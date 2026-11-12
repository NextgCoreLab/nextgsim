//! The exporter against an in-process h2c stub standing in for the NRF (NF
//! management, discovery, the token endpoint) and nwdafd's ingest route.

use std::sync::{Arc, Mutex as StdMutex};

use bytes::{Bytes, BytesMut};
use nextgsim_demo::{DemoRegistry, DemoRuntime, DemoSpec};
use tokio::net::TcpListener;

use super::*;
use crate::congestion::{Status, D441A_MODES};

#[derive(Debug, Clone)]
struct Seen {
    method: String,
    path: String,
    auth: Option<String>,
    body: Bytes,
}

type Handler = Arc<dyn Fn(&Seen, &str) -> (u16, String) + Send + Sync>;

/// One stub on 127.0.0.1:0 answering every request through `handler`, which is
/// also told the stub's own base URI (for the discovery answer).
struct Stub {
    base: String,
    seen: Arc<StdMutex<Vec<Seen>>>,
}

impl Stub {
    async fn start(handler: impl Fn(&Seen, &str) -> (u16, String) + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://{addr}");
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let handler: Handler = Arc::new(handler);
        let (seen2, base2) = (seen.clone(), base.clone());
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                let (handler, seen, base) = (handler.clone(), seen2.clone(), base2.clone());
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(sock).await else {
                        return;
                    };
                    while let Some(Ok((req, mut respond))) = conn.accept().await {
                        let (handler, seen, base) = (handler.clone(), seen.clone(), base.clone());
                        tokio::spawn(async move {
                            let (parts, mut body) = req.into_parts();
                            let mut buf = BytesMut::new();
                            while let Some(Ok(c)) = body.data().await {
                                let _ = body.flow_control().release_capacity(c.len());
                                buf.extend_from_slice(&c);
                            }
                            let s = Seen {
                                method: parts.method.to_string(),
                                path: parts
                                    .uri
                                    .path_and_query()
                                    .map_or_else(String::new, |p| p.as_str().to_string()),
                                auth: parts
                                    .headers
                                    .get("authorization")
                                    .and_then(|v| v.to_str().ok())
                                    .map(str::to_string),
                                body: buf.freeze(),
                            };
                            seen.lock().unwrap().push(s.clone());
                            let (status, body) = handler(&s, &base);
                            let resp = http::Response::builder()
                                .status(status)
                                .header("content-type", "application/json")
                                .body(())
                                .unwrap();
                            if let Ok(mut send) = respond.send_response(resp, false) {
                                let _ = send.send_data(Bytes::from(body), true);
                            }
                        });
                    }
                });
            }
        });
        Self { base, seen }
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }

    fn reports(&self) -> Vec<Seen> {
        self.seen()
            .into_iter()
            .filter(|s| s.path == CELL_REPORTS_PATH)
            .collect()
    }
}

const TOKEN: &str = "tok-d441a";

/// A conformant NRF + NWDAF: registration 201, discovery naming this stub as
/// the NWDAF, a token for `nnwdaf-randata`, and `report_status` for a report.
fn core(report_status: u16) -> impl Fn(&Seen, &str) -> (u16, String) + Send + Sync + 'static {
    move |s, base| {
        if s.path.starts_with("/nnrf-nfm/v1/nf-instances/") {
            (201, "{}".into())
        } else if s.path.starts_with("/nnrf-disc/v1/nf-instances") {
            let (host, port) = base.trim_start_matches("http://").split_once(':').unwrap();
            (
                200,
                json!({"nfInstances": [{"nfType": "NWDAF", "nfServices": [
                    {"serviceName": "nnwdaf-randata",
                     "ipEndPoints": [{"ipv4Address": host, "port": port.parse::<u16>().unwrap()}]}]}]})
                .to_string(),
            )
        } else if s.path == nextgsim_sbi_client::TOKEN_PATH {
            (
                200,
                format!(
                    r#"{{"access_token":"{TOKEN}","token_type":"Bearer","expires_in":3600,"scope":"nnwdaf-randata"}}"#
                ),
            )
        } else if s.path == CELL_REPORTS_PATH {
            (report_status, String::new())
        } else {
            (404, String::new())
        }
    }
}

/// An active D441A demo state in `mode`, leaked for the exporter to hold.
fn demo(mode: &str) -> &'static DemoRuntime {
    let reg = DemoRegistry::new(&[DemoSpec {
        key: D441A_KEY,
        modes: D441A_MODES,
        requires: &[],
    }]);
    let yaml = format!(
        "demo:\n  enabled: true\n  d441a-congestion-loop:\n    enabled: true\n    mode: {mode}\n"
    );
    Box::leak(Box::new(
        DemoRuntime::from_root(&serde_yaml::from_str(&yaml).unwrap(), reg).unwrap(),
    ))
}

fn status() -> &'static std::sync::Mutex<Status> {
    Box::leak(Box::new(std::sync::Mutex::new(Status::default())))
}

fn config(nrf: &str, key_dir: &std::path::Path) -> GnbConfig {
    let mut c = GnbConfig {
        nci: 0x10,
        tac: 1,
        plmn: nextgsim_common::Plmn::new(999, 70, false),
        ..GnbConfig::default()
    };
    c.congestion_export.cell_capacity_mbps = Some(5.0);
    c.congestion_export.nrf_uri = Some(nrf.to_string());
    c.congestion_export.cca_key_file = key_dir.join("cca.key");
    c
}

/// A one-second window serving `kbps`.
fn window(kbps: u64, at_s: u64) -> WindowSample {
    WindowSample {
        end: UNIX_EPOCH + Duration::from_secs(1_800_000_000 + at_s),
        elapsed: Duration::from_secs(1),
        served_octets: kbps * 1000 / 8,
        connected_ues: 2,
        queue_depth_bytes: 1234,
        queue_drops: 3,
    }
}

/// Mode `statistics`: the AF registers at the NRF, discovers the NWDAF, takes a
/// token for NWDAF/nnwdaf-randata, and POSTs the contract body with it.
#[tokio::test]
async fn statistics_mode_posts_the_measured_report_with_a_token() {
    let stub = Stub::start(core(204)).await;
    let dir = tempfile::tempdir().unwrap();
    let st = status();
    let mut ex =
        Exporter::with_demo(config(&stub.base, dir.path()), Some(demo("statistics")), st).unwrap();

    let out = ex.on_window(window(4800, 0)).await;
    assert_eq!(out, Some(Outcome::Status(204)));

    let seen = stub.seen();
    let reg = seen
        .iter()
        .find(|s| s.method == "PUT")
        .expect("NRF registration");
    let profile: Value = serde_json::from_slice(&reg.body).unwrap();
    assert_eq!(profile["nfType"], "AF");
    let disc = seen
        .iter()
        .find(|s| s.path.starts_with("/nnrf-disc/"))
        .expect("discovery");
    assert!(
        disc.path.contains("target-nf-type=NWDAF")
            && disc.path.contains("requester-nf-type=AF")
            && disc.path.contains("service-names=nnwdaf-randata"),
        "{}",
        disc.path
    );
    let token_req = seen
        .iter()
        .find(|s| s.path == nextgsim_sbi_client::TOKEN_PATH)
        .expect("token request");
    let form = String::from_utf8_lossy(&token_req.body);
    assert!(form.contains("scope=nnwdaf-randata"), "{form}");
    assert!(form.contains("targetNfType=NWDAF"), "{form}");

    let reports = stub.reports();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].method, "POST");
    assert_eq!(
        reports[0].auth.as_deref(),
        Some(&*format!("Bearer {TOKEN}"))
    );
    let body: Value = serde_json::from_slice(&reports[0].body).unwrap();
    assert_eq!(
        body,
        json!({
            "ncgi": {"plmnId": {"mcc": "999", "mnc": "70"}, "nrCellId": "000000010"},
            "tai": {"plmnId": {"mcc": "999", "mnc": "70"}, "tac": "000001"},
            "timeStamp": "2027-01-15T08:00:00Z",
            "windowMs": 1000,
            "capacityKbps": 5000,
            "dlServedKbps": 4800,
            "occupancyPct": 96.0,
            "connectedUes": 2,
        })
    );
    let s = st.lock().unwrap().clone();
    assert_eq!(s.exports_accepted, 1);
    assert_eq!(s.last_export.as_deref(), Some("204"));
    assert_eq!((s.queue_depth_bytes, s.queue_drops), (1234, 3));
}

/// Mode `prediction`: no prediction until eight windows exist; then the body
/// carries the OLS extrapolation of the ramp with its R² confidence.
#[tokio::test]
async fn prediction_mode_adds_the_ols_prediction_after_eight_windows() {
    let stub = Stub::start(core(204)).await;
    let dir = tempfile::tempdir().unwrap();
    let mut ex = Exporter::with_demo(
        config(&stub.base, dir.path()),
        Some(demo("prediction")),
        status(),
    )
    .unwrap();
    // 1000, 1250, ... kbps: 20 % + 5 %/s.
    for i in 0..8u64 {
        ex.on_window(window(1000 + 250 * i, i)).await;
    }
    let reports = stub.reports();
    assert_eq!(reports.len(), 8);
    let first: Value = serde_json::from_slice(&reports[6].body).unwrap();
    assert!(first.get("prediction").is_none(), "{first}");
    let last: Value = serde_json::from_slice(&reports[7].body).unwrap();
    // At t=7 the ramp is at 55 %; 5 s later, 80 %.
    assert_eq!(
        last["prediction"],
        json!({"occupancyPct": 80.0, "horizonMs": 5000, "confidence": 100})
    );
}

/// Mode `no-loop`: nothing is sent at all, not even the registration.
#[tokio::test]
async fn no_loop_mode_exports_nothing() {
    let stub = Stub::start(core(204)).await;
    let dir = tempfile::tempdir().unwrap();
    let mut ex = Exporter::with_demo(
        config(&stub.base, dir.path()),
        Some(demo("no-loop")),
        status(),
    )
    .unwrap();
    for i in 0..3 {
        assert_eq!(ex.on_window(window(4800, i)).await, None);
    }
    assert!(stub.seen().is_empty(), "{:?}", stub.seen());
}

/// A runtime switch applies from the next window: statistics posts, the switch
/// to no-loop stops it, and switching back resumes it.
#[tokio::test]
async fn a_mode_switch_applies_from_the_next_window() {
    let stub = Stub::start(core(204)).await;
    let dir = tempfile::tempdir().unwrap();
    let d = demo("statistics");
    let mut ex = Exporter::with_demo(config(&stub.base, dir.path()), Some(d), status()).unwrap();
    ex.on_window(window(4800, 0)).await;
    d.handle_cli("d441a-congestion-loop set mode=no-loop", "test")
        .unwrap();
    ex.on_window(window(4800, 1)).await;
    assert_eq!(stub.reports().len(), 1);
    d.handle_cli("d441a-congestion-loop set mode=statistics", "test")
        .unwrap();
    ex.on_window(window(4800, 2)).await;
    assert_eq!(stub.reports().len(), 2);
}

/// Fail closed: when the NRF refuses the token, no report reaches the NWDAF.
#[tokio::test]
async fn a_refused_token_sends_nothing() {
    let stub = Stub::start(|s, base| {
        if s.path == nextgsim_sbi_client::TOKEN_PATH {
            (400, r#"{"error":"unauthorized_client"}"#.into())
        } else {
            core(204)(s, base)
        }
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let st = status();
    let mut ex =
        Exporter::with_demo(config(&stub.base, dir.path()), Some(demo("statistics")), st).unwrap();
    let out = ex.on_window(window(4800, 0)).await;
    assert!(
        matches!(&out, Some(Outcome::NotSent(why)) if why.contains("no access token")),
        "{out:?}"
    );
    assert!(stub.reports().is_empty(), "nothing reached the NWDAF");
    assert_eq!(st.lock().unwrap().exports_refused, 1);
}

/// The NWDAF's 401 is reported as such and counted refused.
#[tokio::test]
async fn an_nwdaf_401_is_reported() {
    let stub = Stub::start(core(401)).await;
    let dir = tempfile::tempdir().unwrap();
    let st = status();
    let mut ex =
        Exporter::with_demo(config(&stub.base, dir.path()), Some(demo("statistics")), st).unwrap();
    assert_eq!(
        ex.on_window(window(4800, 0)).await,
        Some(Outcome::Status(401))
    );
    assert_eq!(st.lock().unwrap().exports_refused, 1);
}

/// The JWK is published under `<dir>/<nf_instance_id>.jwk` when a dir is set.
#[tokio::test]
async fn the_public_jwk_is_published_when_a_dir_is_set() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = config("http://127.0.0.1:9", dir.path());
    c.congestion_export.cca_public_jwk_dir = Some(dir.path().join("jwks"));
    let _ = Exporter::with_demo(c.clone(), None, status()).unwrap();
    let jwk: Value = serde_json::from_str(
        &std::fs::read_to_string(
            dir.path()
                .join("jwks")
                .join(format!("{}.jwk", c.congestion_export.nf_instance_id)),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(jwk["kid"], json!(c.congestion_export.nf_instance_id));
    assert_eq!(jwk["crv"], "P-256");
}

#[test]
fn identifiers_are_spelled_as_ts_29571() {
    assert_eq!(nr_cell_id(0x10), "000000010");
    assert_eq!(nr_cell_id(0xF_FFFF_FFFF), "FFFFFFFFF");
    let c = GnbConfig {
        plmn: nextgsim_common::Plmn::new(1, 1, true),
        ..GnbConfig::default()
    };
    assert_eq!(plmn_id(&c), json!({"mcc": "001", "mnc": "001"}));
    assert_eq!(rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00Z");
    assert_eq!(
        rfc3339(UNIX_EPOCH + Duration::from_secs(951_782_400)),
        "2000-02-29T00:00:00Z"
    );
}
