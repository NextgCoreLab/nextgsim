//! The D441A cell-report exporter (nextgsim#465): registers at the NRF as an
//! AF, takes an access token for `nnwdaf-randata`, and POSTs one cell report per
//! window to nwdafd's `/nnwdaf-randata/v1/cell-reports`.
//!
//! **Non-normative route.** It stands in for the OAM performance-measurement
//! collection of TS 23.288 §6.2.3 / Table 6.8.2-1 (Radio Resource Utilization);
//! no 3GPP API carries a gNB's cell occupancy to an NWDAF.
//!
//! **Fail closed.** The client always runs with OAuth2: a report either carries
//! an NRF-issued token or is not sent, and the reason is logged. The AF's CCA key
//! comes from `congestion_export.cca_key_file` (created when absent); its public
//! JWK is written to `cca_public_jwk_dir` when that is set, for an NRF that reads
//! its trusted keys from a directory.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nextgsim_common::config::GnbConfig;
use nextgsim_demo::demo_log;
use nextgsim_sbi_client::{CcaKey, OAuth2Config, SbiClient, SbiClientConfig, SbiError, Scope};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::predictor::{self, Prediction};
use super::{update_status, WindowSample, D441A_KEY, MODE_NO_LOOP, MODE_PREDICTION};

/// The token audience and scope of a cell report.
pub const TARGET_NF_TYPE: &str = "NWDAF";
/// The (non-normative) service the report route belongs to.
pub const SERVICE: &str = "nnwdaf-randata";
/// The route, under the NWDAF's API root.
pub const CELL_REPORTS_PATH: &str = "/nnwdaf-randata/v1/cell-reports";
/// The nfType the exporter registers as (TS 23.288 §6.8.1: "The service consumer
/// may be an NF (e.g. NEF, AF, PCF)"; a trusted AF is also an NWDAF data source).
pub const NF_TYPE: &str = "AF";
/// NRF heartbeat timer the exporter registers with; it re-registers every half.
pub const HEARTBEAT_SECS: u64 = 60;

/// `nrCellId`: the 36-bit NCI as 9 hex digits (TS 29.571 `NrCellId`).
pub fn nr_cell_id(nci: u64) -> String {
    format!("{:09X}", nci & 0xF_FFFF_FFFF)
}

/// TS 29.571 `PlmnId` of the gNB.
pub fn plmn_id(config: &GnbConfig) -> Value {
    let p = &config.plmn;
    let mnc = if p.long_mnc {
        format!("{:03}", p.mnc)
    } else {
        format!("{:02}", p.mnc)
    };
    json!({ "mcc": format!("{:03}", p.mcc), "mnc": mnc })
}

/// RFC 3339 UTC, whole seconds, for `t`.
pub fn rfc3339(t: SystemTime) -> String {
    let secs = t
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// The cell report for one window (the wire contract nwdafd's ingest route
/// reads). `prediction` is included only when given.
pub fn report_body(
    config: &GnbConfig,
    sample: &WindowSample,
    prediction: Option<(Prediction, u64)>,
) -> Value {
    let c = &config.congestion_export;
    let capacity = c.cell_capacity_mbps.unwrap_or(0.0);
    let plmn = plmn_id(config);
    let mut body = json!({
        "ncgi": { "plmnId": plmn, "nrCellId": nr_cell_id(config.nci) },
        "tai": { "plmnId": plmn, "tac": format!("{:06X}", config.tac & 0xFF_FFFF) },
        "timeStamp": rfc3339(sample.end),
        "windowMs": sample.elapsed.as_millis() as u64,
        "capacityKbps": (capacity * 1000.0).round() as u64,
        "dlServedKbps": sample.served_kbps().round() as u64,
        "occupancyPct": round1(sample.occupancy_pct(capacity)),
        "connectedUes": sample.connected_ues,
    });
    if let Some((p, horizon_ms)) = prediction {
        body["prediction"] = json!({
            "occupancyPct": round1(p.occupancy_pct),
            "horizonMs": horizon_ms,
            "confidence": p.confidence,
        });
    }
    body
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// The AF profile the exporter registers (TS 29.510 `NFProfile`).
pub fn nf_profile(nf_instance_id: &str) -> Value {
    json!({
        "nfInstanceId": nf_instance_id,
        "nfType": NF_TYPE,
        "nfStatus": "REGISTERED",
        "heartBeatTimer": HEARTBEAT_SECS,
    })
}

/// The first NWDAF endpoint in a TS 29.510 `SearchResult` offering
/// [`SERVICE`]: its first `ipEndPoints` entry (port 7777, nextgcore's SBI port,
/// when it names none), else the instance's first IPv4 address on port 7777.
pub fn nwdaf_from_search_result(result: &Value) -> Option<String> {
    for inst in result.get("nfInstances")?.as_array()? {
        let services = inst
            .get("nfServices")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let svc = services
            .iter()
            .find(|s| s.get("serviceName").and_then(Value::as_str) == Some(SERVICE));
        if let Some(ep) = svc
            .and_then(|s| s.get("ipEndPoints"))
            .and_then(Value::as_array)
            .and_then(|e| e.first())
        {
            if let Some(ip) = ep.get("ipv4Address").and_then(Value::as_str) {
                // nextgcore's SBI port when the profile names none.
                let port = ep.get("port").and_then(Value::as_u64).unwrap_or(7777);
                return Some(format!("http://{ip}:{port}"));
            }
        }
        if let Some(ip) = inst
            .get("ipv4Addresses")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
        {
            return Some(format!("http://{ip}:7777"));
        }
    }
    None
}

/// What one export came to, for the log line and the status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The NWDAF answered with this status.
    Status(u16),
    /// Nothing was sent, and why (no token, no NWDAF, transport).
    NotSent(String),
}

impl Outcome {
    fn label(&self) -> String {
        match self {
            Outcome::Status(s) => s.to_string(),
            Outcome::NotSent(why) => format!("not-sent({why})"),
        }
    }
    fn accepted(&self) -> bool {
        matches!(self, Outcome::Status(s) if (200..300).contains(s))
    }
}

/// The exporter's per-process state.
pub struct Exporter {
    config: GnbConfig,
    nrf: String,
    plain: SbiClient,
    client: Arc<SbiClient>,
    nwdaf: Option<String>,
    registered_at: Option<Instant>,
    history: VecDeque<(f64, f64)>,
    mode: Option<String>,
    started: Instant,
    /// The demo state the mode is read from each window.
    demo: Option<&'static nextgsim_demo::DemoRuntime>,
    /// Where `congestion-status` reads from.
    status: &'static std::sync::Mutex<super::Status>,
}

impl Exporter {
    /// The exporter for `config` (whose block [`super::validate_config`]
    /// accepted). Fails on an unusable CCA key file.
    pub fn new(config: GnbConfig) -> Result<Self, String> {
        Self::with_demo(config, nextgsim_demo::installed(), super::status_cell())
    }

    /// [`Self::new`] reading its mode from `demo` rather than the installed state.
    pub fn with_demo(
        config: GnbConfig,
        demo: Option<&'static nextgsim_demo::DemoRuntime>,
        status: &'static std::sync::Mutex<super::Status>,
    ) -> Result<Self, String> {
        let c = &config.congestion_export;
        let nrf = c
            .nrf_uri
            .clone()
            .ok_or("congestion_export.nrf_uri is not set")?
            .trim_end_matches('/')
            .to_string();
        check_existing_key_file(&c.cca_key_file)?;
        let key = CcaKey::load_or_create(&c.cca_key_file)
            .map_err(|e| format!("CCA key {}: {e}", c.cca_key_file.display()))?;
        if let Some(dir) = &c.cca_public_jwk_dir {
            let path = key
                .publish_jwk(dir, &c.nf_instance_id)
                .map_err(|e| format!("publish the CCA public JWK to {}: {e}", dir.display()))?;
            demo_log!(
                info,
                D441A_KEY,
                "export: CCA public JWK published at {} (kid={})",
                path.display(),
                c.nf_instance_id
            );
        }
        demo_log!(
            info,
            D441A_KEY,
            "export: CCA public JWK {}",
            key.public_jwk(&c.nf_instance_id)
        );
        let base = SbiClientConfig {
            timeout: Duration::from_secs(2),
            retries: 0,
            ..SbiClientConfig::default()
        };
        let client = SbiClient::new(SbiClientConfig {
            oauth2: Some(OAuth2Config::new(
                nrf.clone(),
                c.nf_instance_id.clone(),
                NF_TYPE,
                key,
            )),
            ..base.clone()
        });
        Ok(Self {
            nwdaf: c
                .nwdaf_uri
                .clone()
                .map(|u| u.trim_end_matches('/').to_string()),
            config,
            nrf,
            // NRF management is tokenless in nextgcore (as for sbi-roundtrip).
            plain: SbiClient::new(base),
            client: Arc::new(client),
            registered_at: None,
            history: VecDeque::new(),
            mode: None,
            started: Instant::now(),
            demo,
            status,
        })
    }

    /// Register (or refresh the registration) at the NRF when due.
    async fn keep_registered(&mut self) -> bool {
        let due = self
            .registered_at
            .is_none_or(|t| t.elapsed() >= Duration::from_secs(HEARTBEAT_SECS / 2));
        if !due {
            return true;
        }
        let id = &self.config.congestion_export.nf_instance_id;
        let uri = format!("{}/nnrf-nfm/v1/nf-instances/{id}", self.nrf);
        match self.plain.put_json(&uri, &nf_profile(id), None).await {
            Ok(r) if r.status.is_success() => {
                if self.registered_at.is_none() {
                    demo_log!(
                        info,
                        D441A_KEY,
                        "export: registered at the NRF as {NF_TYPE} {id} (status {})",
                        r.status.as_u16()
                    );
                }
                self.registered_at = Some(Instant::now());
                true
            }
            Ok(r) => {
                demo_log!(
                    warn,
                    D441A_KEY,
                    "export: NRF registration answered {}",
                    r.status.as_u16()
                );
                self.registered_at.is_some()
            }
            Err(e) => {
                demo_log!(warn, D441A_KEY, "export: NRF registration failed: {e}");
                self.registered_at.is_some()
            }
        }
    }

    /// The NWDAF to post to: the configured one, else discovered at the NRF.
    async fn nwdaf(&mut self) -> Result<String, String> {
        if let Some(n) = &self.nwdaf {
            return Ok(n.clone());
        }
        let uri = format!(
            "{}/nnrf-disc/v1/nf-instances?target-nf-type={TARGET_NF_TYPE}&requester-nf-type={NF_TYPE}&service-names={SERVICE}",
            self.nrf
        );
        let v: Value = self
            .plain
            .get_json(&uri, None)
            .await
            .map_err(|e| format!("NWDAF discovery: {e}"))?;
        let n = nwdaf_from_search_result(&v).ok_or("no NWDAF offering nnwdaf-randata")?;
        demo_log!(info, D441A_KEY, "export: discovered NWDAF {n}");
        update_status(self.status, |s| s.nwdaf = Some(n.clone()));
        self.nwdaf = Some(n.clone());
        Ok(n)
    }

    /// Handle one window: record it, and in modes `statistics` / `prediction`
    /// post its report. Returns what was done (`None` in `no-loop`).
    pub async fn on_window(&mut self, sample: WindowSample) -> Option<Outcome> {
        let c = &self.config.congestion_export;
        let capacity = c.cell_capacity_mbps.unwrap_or(0.0);
        let horizon_ms = c.prediction_horizon_ms;
        let occupancy = sample.occupancy_pct(capacity);
        let t = sample
            .end
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or_else(|_| self.started.elapsed().as_secs_f64());
        self.history.push_back((t, occupancy));
        while self.history.len() > predictor::SAMPLES {
            self.history.pop_front();
        }

        let mode = self.demo.and_then(|d| d.mode(D441A_KEY));
        if mode != self.mode {
            match (&self.mode, &mode) {
                (_, Some(m)) if m == MODE_NO_LOOP => {
                    demo_log!(info, D441A_KEY, "export: off (mode=no-loop)")
                }
                (Some(old), Some(new)) => demo_log!(
                    info,
                    D441A_KEY,
                    "export: mode {old} -> {new}, applied from this window"
                ),
                (None, Some(new)) => demo_log!(info, D441A_KEY, "export: on (mode={new})"),
                (_, None) => {}
            }
            self.mode = mode.clone();
        }
        let prediction = match mode.as_deref() {
            Some(MODE_PREDICTION) => {
                let s: Vec<(f64, f64)> = self.history.iter().copied().collect();
                predictor::predict(&s, horizon_ms as f64 / 1000.0).map(|p| (p, horizon_ms))
            }
            _ => None,
        };
        update_status(self.status, |s| {
            s.mode = mode.clone();
            s.capacity_kbps = (capacity * 1000.0).round() as u64;
            s.last_occupancy_pct = Some(round1(occupancy));
            s.last_prediction = prediction.map(|(p, h)| (round1(p.occupancy_pct), p.confidence, h));
            s.queue_depth_bytes = sample.queue_depth_bytes;
            s.queue_drops = sample.queue_drops;
        });
        match mode.as_deref() {
            None | Some(MODE_NO_LOOP) => return None,
            _ => {}
        }

        let body = report_body(&self.config, &sample, prediction);
        let outcome = self.post(&body).await;
        let predicted = prediction
            .map(|(p, h)| {
                format!(
                    " predicted={:.1}% conf={} horizon_ms={h}",
                    p.occupancy_pct, p.confidence
                )
            })
            .unwrap_or_default();
        demo_log!(
            info,
            D441A_KEY,
            "ran-export cell={} occupancy={:.1}% served_kbps={} connected_ues={}{predicted} status={}",
            nr_cell_id(self.config.nci),
            occupancy,
            sample.served_kbps().round() as u64,
            sample.connected_ues,
            outcome.label()
        );
        update_status(self.status, |s| {
            s.last_export = Some(outcome.label());
            if outcome.accepted() {
                s.exports_accepted += 1;
            } else {
                s.exports_refused += 1;
            }
        });
        Some(outcome)
    }

    async fn post(&mut self, body: &Value) -> Outcome {
        if !self.keep_registered().await {
            return Outcome::NotSent("not registered at the NRF".into());
        }
        let nwdaf = match self.nwdaf().await {
            Ok(n) => n,
            Err(e) => return Outcome::NotSent(e),
        };
        let scope = Scope::new(TARGET_NF_TYPE, SERVICE);
        let uri = format!("{nwdaf}{CELL_REPORTS_PATH}");
        match self.client.post_json(&uri, body, Some(&scope)).await {
            Ok(r) => Outcome::Status(r.status.as_u16()),
            Err(SbiError::Unauthorized { .. }) => Outcome::Status(401),
            Err(SbiError::Forbidden) => Outcome::Status(403),
            // Nothing reached the NWDAF without a token: fail closed.
            Err(
                e @ (SbiError::TokenRefused { .. }
                | SbiError::TokenInvalid(_)
                | SbiError::ScopeRequired
                | SbiError::Key(_)),
            ) => Outcome::NotSent(format!("no access token: {e}")),
            Err(e) => {
                // A transport error may mean the NWDAF moved: rediscover next time,
                // unless it was configured.
                if self.config.congestion_export.nwdaf_uri.is_none() {
                    self.nwdaf = None;
                }
                Outcome::NotSent(e.to_string())
            }
        }
    }

    /// Run until the window channel closes.
    pub async fn run(mut self, mut rx: mpsc::Receiver<WindowSample>) {
        while let Some(sample) = rx.recv().await {
            let _ = self.on_window(sample).await;
        }
    }
}

/// Start a built exporter on its own task, fed by the returned sender. Building
/// it (and failing on an unusable key) is startup's job,
/// [`super::prepare_startup`].
pub fn spawn(exporter: Exporter) -> mpsc::Sender<WindowSample> {
    let (tx, rx) = mpsc::channel(16);
    tokio::spawn(exporter.run(rx));
    tx
}

/// A key file that already exists is used only when this process's user owns
/// it and nobody else can read or write it (mode `0600` or tighter): a file
/// another local user planted at the path would otherwise hand them the AF's
/// private key. A file that does not exist is created `0600` by
/// `CcaKey::load_or_create`.
pub fn check_existing_key_file(path: &std::path::Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = match std::fs::symlink_metadata(path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(format!("CCA key {}: {e}", path.display())),
        };
        if !meta.file_type().is_file() {
            return Err(format!(
                "CCA key {} is not a regular file; refusing it",
                path.display()
            ));
        }
        if meta.mode() & 0o077 != 0 {
            return Err(format!(
                "CCA key {} has mode {:o}; it must be 0600 (owner only)",
                path.display(),
                meta.mode() & 0o777
            ));
        }
        // /proc/self is owned by the process's effective uid (Linux). Elsewhere
        // the owner is not checked; the mode still is.
        if let Ok(me) = std::fs::metadata("/proc/self") {
            if meta.uid() != me.uid() {
                return Err(format!(
                    "CCA key {} is owned by uid {}, not by this process (uid {}); refusing it",
                    path.display(),
                    meta.uid(),
                    me.uid()
                ));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests;
