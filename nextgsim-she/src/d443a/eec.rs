//! The UE's EEC client (TS 24.558 `Eees_EECRegistration`) and the AC's job.
//!
//! For each job: an EEC registration with `easSelReqInd: true` and the AC
//! profile, so the EES selects (§5.2.2.2 step 1 iii) B)); the registration is
//! deleted again at once (it exists only to ask). Then:
//! - `201` with `discoveredEas`: the job goes to the first EAS, and the AC logs
//!   `eas=<id> rtt=<ms> predicted=<ms>` with `result=ok`, or `deadline-miss`
//!   when the answer took longer than the UE's deadline (or never came);
//! - `404` (no EAS within the EES's policy): `on-device fallback reason=ees-404`;
//! - anything else (401, unreachable, a 201 with no selection): on-device too,
//!   with the reason.
//!
//! Every socket (the EES's, the NRF token request's and the job's) is bound
//! to the configured device, the PDU session's TUN.

use std::time::Duration;

use nextgsim_common::config::EecClientConfig;
use nextgsim_demo::demo_log;
use nextgsim_sbi_client::{SbiClient, SbiError, Scope};
use serde_json::Value;

use super::eel::{self, Selected};
use super::{job, EECREG_PATH, EES_NF_TYPE, KEY, SCOPE_EECREGISTRATION};

/// How one job ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The job ran on the selected EAS.
    Offloaded {
        eas_id: String,
        rtt_ms: u32,
        predicted_ms: Option<u32>,
        budget_src: Option<String>,
        /// The answer came within the UE's deadline.
        in_time: bool,
    },
    /// The job was sent to the selected EAS and no answer came.
    NoAnswer { eas_id: String, why: String },
    /// No EAS: the job runs on the device.
    OnDevice { reason: String },
}

impl Outcome {
    /// The log line of job `job` (after the `[demo:<key>]` prefix).
    pub fn line(&self, job: u32, deadline_ms: u32, mode: &str) -> String {
        match self {
            Self::Offloaded {
                eas_id,
                rtt_ms,
                predicted_ms,
                budget_src,
                in_time,
            } => format!(
                "JOB job={job} eas={eas_id} rtt={rtt_ms} predicted={} src={} deadline={deadline_ms} \
                 mode={mode} result={}",
                predicted_ms.map_or_else(|| "?".to_string(), |p| p.to_string()),
                budget_src.as_deref().unwrap_or("?"),
                if *in_time { "ok" } else { "deadline-miss" }
            ),
            Self::NoAnswer { eas_id, why } => format!(
                "JOB job={job} eas={eas_id} rtt=? deadline={deadline_ms} mode={mode} \
                 result=deadline-miss ({why})"
            ),
            Self::OnDevice { reason } => {
                format!("JOB job={job} on-device fallback reason={reason} mode={mode}")
            }
        }
    }

    /// Whether this job missed the UE's deadline on an EAS.
    pub fn missed(&self) -> bool {
        matches!(
            self,
            Self::NoAnswer { .. } | Self::Offloaded { in_time: false, .. }
        )
    }
}

fn scope() -> Scope {
    Scope::new(EES_NF_TYPE, SCOPE_EECREGISTRATION)
}

/// Ask the EES for a selection: the `discoveredEas`, or the on-device reason.
pub async fn select(client: &SbiClient, cfg: &EecClientConfig) -> Result<Vec<Selected>, String> {
    let api_root = cfg.ees.trim_end_matches('/');
    let body = eel::eec_registration(cfg);
    let r = client
        .post_json(&format!("{api_root}{EECREG_PATH}"), &body, Some(&scope()))
        .await
        .map_err(|e| match e {
            SbiError::Unauthorized { .. } => "ees-401".to_string(),
            SbiError::Forbidden => "ees-403".to_string(),
            SbiError::Connect { .. } | SbiError::Timeout(_) => "ees-unreachable".to_string(),
            other => format!("ees-error ({other})"),
        })?;
    let status = r.status.as_u16();
    if status != 201 {
        return Err(format!("ees-{status}"));
    }
    // The registration existed only to ask; leave nothing behind.
    if let Some(loc) = r.headers.get("location").and_then(|v| v.to_str().ok()) {
        let url = if loc.starts_with("http://") {
            loc.to_string()
        } else {
            format!("{api_root}{loc}")
        };
        if let Err(e) = client.delete(&url, Some(&scope())).await {
            demo_log!(debug, KEY, "EEC deregistration {url}: {e}");
        }
    }
    let v: Value = serde_json::from_slice(&r.body).map_err(|_| "ees-bad-body".to_string())?;
    let sel = eel::selection(&v);
    if sel.is_empty() {
        return Err("no-discovered-eas".into());
    }
    Ok(sel)
}

/// Run job `job`: select, then offload to the first EAS or fall back.
pub async fn offload(client: &SbiClient, cfg: &EecClientConfig, job: u32) -> Outcome {
    let sel = match select(client, cfg).await {
        Ok(s) => s,
        Err(reason) => return Outcome::OnDevice { reason },
    };
    let first = &sel[0];
    let Some(ip) = first.ip else {
        return Outcome::OnDevice {
            reason: "eas-without-address".into(),
        };
    };
    let dst = std::net::SocketAddr::new(ip, cfg.job_port);
    let timeout = Duration::from_millis(u64::from(cfg.job_timeout_ms.max(1)));
    match job::run(dst, cfg.device.as_deref(), job, cfg.req_comp, timeout).await {
        Ok(a) => Outcome::Offloaded {
            eas_id: first.eas_id.clone(),
            rtt_ms: a.rtt_ms,
            predicted_ms: first.achievable_latency_ms,
            budget_src: first.budget_src.clone(),
            in_time: a.rtt_ms <= cfg.deadline_ms,
        },
        Err(why) => Outcome::NoAnswer {
            eas_id: first.eas_id.clone(),
            why,
        },
    }
}

/// Refuse an `eec:` block the client cannot run.
pub fn validate(cfg: &EecClientConfig) -> Result<(), String> {
    if cfg.eec_id.trim().is_empty() {
        return Err("eec.eec_id is empty".into());
    }
    if cfg.ac_id.trim().is_empty() {
        return Err("eec.ac_id is empty".into());
    }
    if cfg.eass.is_empty() || cfg.eass.iter().any(|e| e.trim().is_empty()) {
        return Err("eec.eass must name at least one easId".into());
    }
    if !cfg.ees.starts_with("http://") {
        return Err("eec.ees must be http://host:port (h2c)".into());
    }
    if cfg.deadline_ms == 0 {
        return Err("eec.deadline_ms must be positive".into());
    }
    Ok(())
}
