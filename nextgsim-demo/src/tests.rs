use std::io::Write;
use std::sync::{Arc, Mutex};

use super::*;

const CONGESTION: DemoSpec = DemoSpec {
    key: "d441a-congestion-loop",
    modes: &["baseline", "closed-loop"],
    requires: &[],
};
const EDGE: DemoSpec = DemoSpec {
    key: "edge-select",
    modes: &["delay", "compute"],
    requires: &["d441a-congestion-loop"],
};

fn registry() -> DemoRegistry {
    DemoRegistry::new(&[CONGESTION, EDGE])
}

fn root(yaml: &str) -> Value {
    serde_yaml::from_str(yaml).expect("test YAML parses")
}

fn validate(yaml: &str) -> Result<ActiveDemos, DemoConfigError> {
    DemoConfig::from_root(&root(yaml))?.validate(&registry())
}

fn runtime(yaml: &str) -> DemoRuntime {
    DemoRuntime::from_root(&root(yaml), registry()).expect("valid demo config")
}

const ON: &str = "
demo:
  enabled: true
  d441a-congestion-loop:
    enabled: true
    mode: baseline
";

/// A writer the test subscriber logs into, so a test can assert the exact lines.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Capture {
    fn lines(&self) -> Vec<String> {
        String::from_utf8(self.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }
}

/// Run `f` with a thread-local subscriber capturing `LEVEL message` lines. The
/// subscriber is scoped to this thread, so parallel tests do not see each
/// other's lines.
fn captured<R>(f: impl FnOnce() -> R) -> (R, Vec<String>) {
    let cap = Capture::default();
    let writer = cap.clone();
    let sub = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .with_max_level(tracing::Level::TRACE)
        .finish();
    let r = tracing::subscriber::with_default(sub, f);
    (r, cap.lines())
}

// ---- schema and errors --------------------------------------------------

#[test]
fn absent_null_and_empty_blocks_mean_no_demo() {
    for yaml in [
        "gnb: 1\n",
        "demo: ~\n",
        "demo: {}\n",
        "demo:\n  enabled: false\n",
    ] {
        let a = validate(yaml).unwrap();
        assert!(a.is_empty(), "{yaml:?}");
        assert_eq!(a.banner(), None);
    }
}

#[test]
fn an_enabled_demo_under_the_kill_switch_is_active() {
    let a = validate(ON).unwrap();
    assert_eq!(
        a.entries().collect::<Vec<_>>(),
        vec![("d441a-congestion-loop", "baseline")]
    );
    assert!(a.is_active("d441a-congestion-loop"));
    assert_eq!(
        a.banner().as_deref(),
        Some("DEMO MODE ACTIVE: d441a-congestion-loop=baseline")
    );
}

#[test]
fn the_kill_switch_overrides_every_demo() {
    for kill in ["false", "~"] {
        let yaml = format!(
            "demo:\n  enabled: {kill}\n  d441a-congestion-loop:\n    enabled: true\n    mode: baseline\n"
        );
        assert!(validate(&yaml).unwrap().is_empty(), "enabled: {kill}");
    }
    // An omitted kill switch defaults to off as well.
    let yaml = "demo:\n  d441a-congestion-loop:\n    enabled: true\n    mode: baseline\n";
    assert!(validate(yaml).unwrap().is_empty());
}

#[test]
fn a_disabled_entry_is_not_active() {
    let yaml = "demo:\n  enabled: true\n  d441a-congestion-loop:\n    enabled: false\n    mode: baseline\n";
    assert!(validate(yaml).unwrap().is_empty());
}

#[test]
fn an_unknown_key_fails_even_with_the_kill_switch_off() {
    for yaml in [
        "demo:\n  enabled: true\n  d441a-congestion-lop:\n    enabled: true\n    mode: baseline\n",
        "demo:\n  enabled: false\n  d441a-congestion-lop: {}\n",
    ] {
        assert_eq!(
            validate(yaml),
            Err(DemoConfigError::UnknownKey {
                key: "d441a-congestion-lop".into(),
                known: vec!["d441a-congestion-loop", "edge-select"],
            })
        );
    }
}

#[test]
fn a_default_build_registers_nothing_and_rejects_every_key() {
    let empty = DemoRegistry::default();
    let err = DemoConfig::from_root(&root(ON))
        .unwrap()
        .validate(&empty)
        .unwrap_err();
    assert_eq!(
        err,
        DemoConfigError::UnknownKey {
            key: "d441a-congestion-loop".into(),
            known: vec![],
        }
    );
}

#[test]
fn an_unknown_mode_fails_even_on_a_disabled_entry() {
    for enabled in ["true", "false"] {
        let yaml = format!(
            "demo:\n  enabled: false\n  d441a-congestion-loop:\n    enabled: {enabled}\n    mode: closed-lop\n"
        );
        assert_eq!(
            validate(&yaml),
            Err(DemoConfigError::UnknownMode {
                demo: "d441a-congestion-loop".into(),
                mode: "closed-lop".into(),
                known: vec!["baseline", "closed-loop"],
            }),
            "enabled: {enabled}"
        );
    }
}

#[test]
fn a_missing_dependency_fails() {
    // Absent.
    let yaml = "demo:\n  enabled: true\n  edge-select:\n    enabled: true\n    mode: delay\n";
    let want = Err(DemoConfigError::MissingDependency {
        demo: "edge-select".into(),
        requires: "d441a-congestion-loop".into(),
    });
    assert_eq!(validate(yaml), want);
    // Present but disabled.
    let yaml = "demo:\n  enabled: true\n  edge-select:\n    enabled: true\n    mode: delay\n  d441a-congestion-loop:\n    enabled: false\n    mode: baseline\n";
    assert_eq!(validate(yaml), want);
    // Present and enabled.
    let yaml = "demo:\n  enabled: true\n  edge-select:\n    enabled: true\n    mode: delay\n  d441a-congestion-loop:\n    enabled: true\n    mode: baseline\n";
    assert_eq!(
        validate(yaml)
            .unwrap()
            .entries()
            .map(|(k, _)| k)
            .collect::<Vec<_>>(),
        vec!["d441a-congestion-loop", "edge-select"]
    );
}

#[test]
fn malformed_blocks_fail_by_name() {
    let cases = [
        ("demo: true\n", "demo must be a mapping"),
        (
            "demo:\n  enabled: yes-please\n",
            "demo.enabled must be a boolean",
        ),
        (
            "demo:\n  d441a-congestion-loop: on\n",
            "demo.d441a-congestion-loop must be a mapping",
        ),
        (
            "demo:\n  d441a-congestion-loop:\n    enabled: 1\n",
            "demo.d441a-congestion-loop.enabled must be a boolean",
        ),
        (
            "demo:\n  d441a-congestion-loop:\n    mode: 3\n",
            "demo.d441a-congestion-loop.mode must be a string",
        ),
        (
            "demo:\n  d441a-congestion-loop:\n    enabeld: true\n",
            "demo.d441a-congestion-loop has unknown field \"enabeld\"",
        ),
        (
            "demo:\n  enabled: true\n  d441a-congestion-loop:\n    enabled: true\n",
            "demo.d441a-congestion-loop.mode is required",
        ),
    ];
    for (yaml, want) in cases {
        match validate(yaml) {
            Err(DemoConfigError::Malformed { detail }) => {
                assert!(detail.contains(want), "{yaml:?}: {detail}");
            }
            other => panic!("{yaml:?}: expected Malformed, got {other:?}"),
        }
    }
}

#[test]
#[should_panic(expected = "requires unregistered demo")]
fn a_registry_naming_an_unregistered_dependency_is_a_programming_error() {
    let _ = DemoRegistry::new(&[EDGE]);
}

// ---- visibility ---------------------------------------------------------

#[test]
fn no_metric_family_at_all_when_off() {
    let mut out = String::new();
    validate("demo:\n  enabled: false\n  d441a-congestion-loop:\n    enabled: true\n    mode: baseline\n")
        .unwrap()
        .render_metrics(&mut out);
    assert_eq!(out, "");
    runtime("gnb: 1\n").render_metrics(&mut out);
    assert_eq!(out, "");
}

#[test]
fn the_metric_family_names_each_active_demo_and_mode() {
    let mut out = String::new();
    runtime(ON).render_metrics(&mut out);
    assert_eq!(
        out,
        "# HELP nextg_demo_enabled 1 for each demo active in this process (nextgsim#458)\n\
         # TYPE nextg_demo_enabled gauge\n\
         nextg_demo_enabled{demo=\"d441a-congestion-loop\",mode=\"baseline\"} 1\n"
    );
}

#[test]
fn the_log_helper_prefixes_the_key() {
    assert_eq!(
        prefix("d441a-congestion-loop"),
        "[demo:d441a-congestion-loop]"
    );
    let ((), lines) = captured(|| demo_log!(info, "d441a-congestion-loop", "cap {}", 3));
    assert_eq!(lines, vec![" INFO [demo:d441a-congestion-loop] cap 3"]);
}

// ---- runtime switching (nr-cli) ----------------------------------------

#[test]
fn set_switches_an_active_demo_applies_to_the_next_read_and_is_audited() {
    let rt = runtime(ON);
    assert_eq!(
        rt.mode("d441a-congestion-loop").as_deref(),
        Some("baseline")
    );
    let (out, lines) =
        captured(|| rt.handle_cli("d441a-congestion-loop set mode=closed-loop", "nr-cli"));
    assert_eq!(
        out.unwrap(),
        "demo=d441a-congestion-loop mode=closed-loop (was baseline; applies to the next decision)"
    );
    // The next decision reads the new mode, and the metric follows it.
    assert_eq!(
        rt.mode("d441a-congestion-loop").as_deref(),
        Some("closed-loop")
    );
    let mut m = String::new();
    rt.render_metrics(&mut m);
    assert!(m.contains("nextg_demo_enabled{demo=\"d441a-congestion-loop\",mode=\"closed-loop\"} 1"));
    assert!(!m.contains("baseline"));
    assert_eq!(
        lines,
        vec![
            " WARN [demo:d441a-congestion-loop] audit: mode baseline -> closed-loop from nr-cli; \
             applies to the next decision",
            " WARN DEMO MODE ACTIVE: d441a-congestion-loop=closed-loop",
        ]
    );
}

#[test]
fn a_bogus_mode_is_rejected_audited_and_leaves_the_mode_unchanged() {
    let rt = runtime(ON);
    let (out, lines) = captured(|| rt.handle_cli("d441a-congestion-loop set mode=bogus", "nr-cli"));
    assert_eq!(
        out,
        Err(DemoCliError::Config(DemoConfigError::UnknownMode {
            demo: "d441a-congestion-loop".into(),
            mode: "bogus".into(),
            known: vec!["baseline", "closed-loop"],
        }))
    );
    assert_eq!(
        rt.mode("d441a-congestion-loop").as_deref(),
        Some("baseline")
    );
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines[0].starts_with(
            " WARN [demo:d441a-congestion-loop] audit: rejected set mode=\"bogus\" from nr-cli: "
        ),
        "{lines:?}"
    );
    assert!(lines[0].ends_with("mode unchanged (baseline)"), "{lines:?}");
}

#[test]
fn set_on_an_unknown_key_or_an_inactive_demo_is_rejected_and_audited() {
    let rt = runtime(ON);
    let (out, lines) = captured(|| rt.handle_cli("no-such-demo set mode=baseline", "nr-cli"));
    assert!(matches!(
        out,
        Err(DemoCliError::Config(DemoConfigError::UnknownKey { .. }))
    ));
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with(" WARN [demo] audit: rejected set on demo \"no-such-demo\""));

    // edge-select is registered but the config did not enable it: `set` does not
    // enable a demo.
    let (out, lines) = captured(|| rt.handle_cli("edge-select set mode=delay", "nr-cli"));
    assert_eq!(
        out,
        Err(DemoCliError::NotActive {
            demo: "edge-select".into()
        })
    );
    assert_eq!(rt.mode("edge-select"), None);
    assert!(lines[0].contains("[demo:edge-select] audit: rejected set"));
    assert!(
        lines[0].ends_with("mode unchanged (not active)"),
        "{lines:?}"
    );
}

#[test]
fn set_to_the_current_mode_is_audited_as_unchanged() {
    let rt = runtime(ON);
    let (out, lines) =
        captured(|| rt.handle_cli("d441a-congestion-loop set mode=baseline", "nr-cli"));
    assert_eq!(
        out.unwrap(),
        "demo=d441a-congestion-loop mode=baseline (unchanged)"
    );
    assert_eq!(
        lines,
        vec![
            " WARN [demo:d441a-congestion-loop] audit: set mode=baseline from nr-cli: already \
              baseline, unchanged"
        ]
    );
}

#[test]
fn show_reports_each_demo_and_the_kill_switch() {
    let rt = runtime(ON);
    assert_eq!(
        rt.handle_cli("d441a-congestion-loop show", "nr-cli").unwrap(),
        "demo=d441a-congestion-loop active=true mode=baseline modes=baseline,closed-loop requires=-"
    );
    assert_eq!(
        rt.handle_cli("show", "nr-cli").unwrap(),
        "demo.enabled=true\n\
         demo=d441a-congestion-loop active=true mode=baseline modes=baseline,closed-loop requires=-\n\
         demo=edge-select active=false mode=- modes=delay,compute requires=d441a-congestion-loop"
    );
    assert!(matches!(
        rt.handle_cli("nope show", "nr-cli"),
        Err(DemoCliError::Config(DemoConfigError::UnknownKey { .. }))
    ));
    let empty = DemoRuntime::default();
    assert!(empty
        .handle_cli("show", "nr-cli")
        .unwrap()
        .contains("no demo is registered in this binary"));
}

#[test]
fn metrics_command_is_empty_when_off() {
    assert_eq!(
        runtime("gnb: 1\n").handle_cli("metrics", "nr-cli").unwrap(),
        ""
    );
    assert!(runtime(ON)
        .handle_cli("metrics", "nr-cli")
        .unwrap()
        .contains("nextg_demo_enabled{demo=\"d441a-congestion-loop\",mode=\"baseline\"} 1"));
}

#[test]
fn malformed_cli_lines_are_usage_errors_and_change_nothing() {
    let rt = runtime(ON);
    for args in [
        "",
        "d441a-congestion-loop",
        "d441a-congestion-loop set",
        "d441a-congestion-loop set closed-loop",
        "d441a-congestion-loop set mode=closed-loop extra",
    ] {
        assert!(
            matches!(rt.handle_cli(args, "nr-cli"), Err(DemoCliError::Usage(_))),
            "{args:?}"
        );
    }
    assert_eq!(
        rt.mode("d441a-congestion-loop").as_deref(),
        Some("baseline")
    );
}

#[test]
fn route_cli_claims_only_demo_commands() {
    assert!(route_cli("status", "nr-cli").is_none());
    assert!(route_cli("demolish", "nr-cli").is_none());
    assert!(route_cli("ue-list", "nr-cli").is_none());
    // INSTALLED is never set in this crate's tests, so a demo command reports it.
    assert_eq!(
        route_cli("demo show", "nr-cli"),
        Some(Err(DemoCliError::NotInstalled))
    );
    assert_eq!(
        route_cli("  DEMO show", "nr-cli"),
        Some(Err(DemoCliError::NotInstalled))
    );
}
