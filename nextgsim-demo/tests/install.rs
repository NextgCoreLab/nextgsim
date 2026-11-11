//! The process-global install path. Its own test binary (so its own process):
//! the unit tests assert the never-installed state, and `INSTALLED` is
//! write-once, so the two cannot share a process. One test, so nothing here
//! races on the global either.

use std::io::Write;
use std::sync::{Arc, Mutex};

use nextgsim_demo::{
    active_mode, demo_log, init_from_yaml, install, installed, render_installed_metrics, route_cli,
    AlreadyInstalled, DemoRegistry, DemoRuntime, DemoSpec,
};

const CONGESTION: DemoSpec = DemoSpec {
    key: "d441a-congestion-loop",
    modes: &["baseline", "closed-loop"],
    requires: &[],
};

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

#[test]
fn install_logs_one_banner_is_write_once_and_serves_the_cli() {
    assert!(installed().is_none());
    let mut out = String::new();
    render_installed_metrics(&mut out);
    assert_eq!(out, "", "nothing before install");

    let cap = Capture::default();
    let writer = cap.clone();
    let sub = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .finish();
    tracing::subscriber::with_default(sub, || {
        let yaml = "gnb: 1\ndemo:\n  enabled: true\n  d441a-congestion-loop:\n    enabled: true\n    mode: baseline\n";
        let rt = init_from_yaml(yaml, DemoRegistry::new(&[CONGESTION])).expect("installs");
        assert_eq!(
            rt.mode("d441a-congestion-loop").as_deref(),
            Some("baseline")
        );

        // Write-once: a second init or install is refused and changes nothing.
        assert_eq!(
            init_from_yaml("demo: ~\n", DemoRegistry::default()).err(),
            Some(nextgsim_demo::DemoConfigError::AlreadyInstalled)
        );
        assert_eq!(
            install(DemoRuntime::default()).err(),
            Some(AlreadyInstalled)
        );
        assert_eq!(
            active_mode("d441a-congestion-loop").as_deref(),
            Some("baseline")
        );

        // The node's CLI path reaches the installed state.
        assert_eq!(
            route_cli("demo d441a-congestion-loop set mode=closed-loop", "nr-cli")
                .unwrap()
                .unwrap(),
            "demo=d441a-congestion-loop mode=closed-loop (was baseline; applies to the next decision)"
        );
        assert_eq!(
            active_mode("d441a-congestion-loop").as_deref(),
            Some("closed-loop")
        );
        demo_log!(
            info,
            "d441a-congestion-loop",
            "next decision uses {}",
            "closed-loop"
        );
    });

    let mut out = String::new();
    render_installed_metrics(&mut out);
    assert!(
        out.contains("nextg_demo_enabled{demo=\"d441a-congestion-loop\",mode=\"closed-loop\"} 1")
    );

    let lines: Vec<String> = String::from_utf8(cap.0.lock().unwrap().clone())
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(
        lines,
        vec![
            " WARN DEMO MODE ACTIVE: d441a-congestion-loop=baseline",
            " WARN [demo:d441a-congestion-loop] audit: mode baseline -> closed-loop from nr-cli; \
             applies to the next decision",
            " WARN DEMO MODE ACTIVE: d441a-congestion-loop=closed-loop",
            " INFO [demo:d441a-congestion-loop] next decision uses closed-loop",
        ]
    );
}
