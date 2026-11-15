//! The `nr-she-eas` EAS sidecar: its config, and its EDGE-3 registration at
//! the EES (TS 29.558 `Eees_EASRegistration`).
//!
//! ```yaml
//! eas_id: eas-local
//! address: 10.60.0.11        # the dn (or pod) address; default 127.0.0.1; never published
//! max_resp_time_ms: 10       # svcKpi.maxRespTime
//! avl_comp: 50               # svcKpi.avlComp
//! ees:
//!   api_root: "http://172.23.0.31:7777"
//!   oauth2: { nrf: "http://172.23.0.10:7777", nf_instance_id: "...",
//!             cca_key_file: /var/lib/eas/cca.pem, register: true }
//! demo:
//!   enabled: true
//!   d443a-edge-budget: { enabled: true, mode: measured-budget }
//! ```
//!
//! The registration is made once and then left alone: in Acts 2 and 3 the
//! EAS's advertisement stays byte-identical while the path changes under it.
//! It is checked every `check_interval_s`, and re-created if the EES lost it
//! (an eesd restart). Applied load lowers the local `avlComp` (the jobs slow
//! down) and, unless `--no-kpi-patch`, merge-patches `svcKpi.avlComp` at the
//! EES (`ModifyIndEASRegistration`). On shutdown the registration is deleted,
//! so a restarted sidecar does not leave a stale twin under its `easId`.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use nextgsim_common::config::NrfOAuth2Config;
use nextgsim_demo::demo_log;
use nextgsim_sbi_client::{SbiClient, SbiError, Scope};
use serde::Deserialize;
use serde_json::Value;

use super::eel::{self, EasAd};
use super::{EASREG_PATH, EES_NF_TYPE, KEY, SCOPE_EASREGISTRATION};

/// The sidecar's config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EasConfig {
    /// `easProf.easId`.
    pub eas_id: String,
    /// The address the reflector and the job listener bind, and the EAS's
    /// `endPt`. Default loopback: set it to the `dn` (or pod) address.
    #[serde(default = "default_address")]
    pub address: IpAddr,
    /// The STAMP Session-Reflector port.
    #[serde(default = "default_stamp_port")]
    pub stamp_port: u16,
    /// The job listener port.
    #[serde(default = "default_job_port")]
    pub job_port: u16,
    /// `svcKpi.maxRespTime`, ms.
    pub max_resp_time_ms: u32,
    /// `svcKpi.avlComp`, unloaded.
    pub avl_comp: u32,
    /// `easProf.acIds`, if any.
    #[serde(default)]
    pub ac_ids: Vec<String>,
    /// The EES.
    pub ees: EesConfig,
    /// How often the registration is checked, s.
    #[serde(default = "default_check_interval")]
    pub check_interval_s: u64,
    /// The `demo:` block (read by `nextgsim-demo`, accepted here).
    #[serde(default)]
    pub demo: Option<serde_yaml::Value>,
}

/// `ees:`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EesConfig {
    /// `http://host:port` (h2c).
    pub api_root: String,
    /// NRF OAuth2. Absent: calls carry no token, which eesd refuses (401).
    #[serde(default)]
    pub oauth2: Option<NrfOAuth2Config>,
}

fn default_address() -> IpAddr {
    IpAddr::V4(Ipv4Addr::LOCALHOST)
}
fn default_stamp_port() -> u16 {
    super::stamp::STAMP_PORT
}
fn default_job_port() -> u16 {
    super::job::DEFAULT_PORT
}
fn default_check_interval() -> u64 {
    5
}

impl EasConfig {
    /// Refuse a config the sidecar cannot serve.
    pub fn validate(&self) -> Result<(), String> {
        if self.eas_id.trim().is_empty() {
            return Err("eas_id is empty".into());
        }
        if self.eas_id.len() > super::job::MAX_EAS_ID {
            return Err(format!(
                "eas_id is longer than {} octets",
                super::job::MAX_EAS_ID
            ));
        }
        if self.avl_comp == 0 {
            return Err("avl_comp must be positive".into());
        }
        if self.address.is_unspecified() {
            return Err(
                "address must be the EAS's own (dn or pod) address: it is also the endPt \
                 the EES hands out, so a wildcard cannot be it"
                    .into(),
            );
        }
        if !self.ees.api_root.starts_with("http://") {
            return Err("ees.api_root must be http://host:port (h2c)".into());
        }
        Ok(())
    }

    /// The advertisement.
    pub fn ad(&self) -> EasAd {
        EasAd {
            eas_id: self.eas_id.clone(),
            ip: self.address,
            max_resp_time_ms: self.max_resp_time_ms,
            avl_comp: self.avl_comp,
            ac_ids: self.ac_ids.clone(),
        }
    }

    pub fn stamp_addr(&self) -> SocketAddr {
        SocketAddr::new(self.address, self.stamp_port)
    }

    pub fn job_addr(&self) -> SocketAddr {
        SocketAddr::new(self.address, self.job_port)
    }
}

fn scope() -> Scope {
    Scope::new(EES_NF_TYPE, SCOPE_EASREGISTRATION)
}

/// The EAS's registration at the EES.
#[derive(Debug)]
pub struct Registrar {
    client: SbiClient,
    api_root: String,
    ad: EasAd,
    /// The Individual EAS Registration resource, absolute.
    location: Option<String>,
}

/// What a check found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Check {
    /// Registered just now (a fresh POST).
    Registered(String),
    /// The registration is still there.
    Alive,
    /// The EES could not be reached or refused; retried next time.
    Failed(String),
}

impl Registrar {
    pub fn new(client: SbiClient, api_root: &str, ad: EasAd) -> Self {
        Self {
            client,
            api_root: api_root.trim_end_matches('/').to_string(),
            ad,
            location: None,
        }
    }

    /// The registration resource, if any.
    pub fn location(&self) -> Option<&str> {
        self.location.as_deref()
    }

    /// What is advertised now.
    pub fn ad(&self) -> &EasAd {
        &self.ad
    }

    fn absolute(&self, loc: &str) -> String {
        if loc.starts_with("http://") || loc.starts_with("https://") {
            loc.to_string()
        } else {
            format!("{}{loc}", self.api_root)
        }
    }

    /// Register if not registered; otherwise check the registration and
    /// re-register if the EES no longer has it (404).
    pub async fn ensure(&mut self) -> Check {
        if let Some(loc) = self.location.clone() {
            match self.client.get_json::<Value>(&loc, Some(&scope())).await {
                Ok(_) => return Check::Alive,
                Err(SbiError::Status { status: 404 }) => {
                    demo_log!(
                        warn,
                        KEY,
                        "EAS-REG eas={} registration {loc} is gone at the EES (restarted?): \
                         registering again",
                        self.ad.eas_id
                    );
                    self.location = None;
                }
                Err(e) => return Check::Failed(format!("check: {e}")),
            }
        }
        let body = eel::eas_registration(&self.ad);
        let url = format!("{}{EASREG_PATH}", self.api_root);
        match self.client.post_json(&url, &body, Some(&scope())).await {
            Ok(r) if r.status.as_u16() == 201 => {
                let Some(loc) = r
                    .headers
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                    .map(|l| self.absolute(l))
                else {
                    return Check::Failed("201 without a Location".into());
                };
                let id = serde_json::from_slice::<Value>(&r.body)
                    .ok()
                    .and_then(|v| v["registrationId"].as_str().map(str::to_string))
                    .unwrap_or_else(|| loc.rsplit('/').next().unwrap_or("").to_string());
                demo_log!(
                    info,
                    KEY,
                    "EAS-REG eas={} registrationId={id} endPt={} maxRespTime={} ms avlComp={}",
                    self.ad.eas_id,
                    self.ad.ip,
                    self.ad.max_resp_time_ms,
                    self.ad.avl_comp
                );
                self.location = Some(loc.clone());
                Check::Registered(loc)
            }
            Ok(r) => Check::Failed(format!("register: status {}", r.status.as_u16())),
            Err(e) => Check::Failed(format!("register: {e}")),
        }
    }

    /// Merge-patch `svcKpi.avlComp` to `avl_comp`. On success the
    /// advertisement (and so a later re-registration) carries it.
    pub async fn patch_avl_comp(&mut self, avl_comp: u32) -> Result<(), String> {
        let Some(loc) = self.location.clone() else {
            return Err("not registered".into());
        };
        let body = eel::avl_comp_patch(&self.ad, avl_comp);
        let r = self
            .client
            .patch_merge_json(&loc, &body, Some(&scope()))
            .await
            .map_err(|e| e.to_string())?;
        let status = r.status.as_u16();
        if status != 200 && status != 204 {
            return Err(format!("status {status}"));
        }
        demo_log!(
            info,
            KEY,
            "EAS-KPI-PATCH eas={} svcKpi.avlComp {} -> {avl_comp} ({status})",
            self.ad.eas_id,
            self.ad.avl_comp
        );
        self.ad.avl_comp = avl_comp;
        Ok(())
    }

    /// Delete the registration.
    pub async fn deregister(&mut self) -> Result<(), String> {
        let Some(loc) = self.location.take() else {
            return Ok(());
        };
        let r = self
            .client
            .delete(&loc, Some(&scope()))
            .await
            .map_err(|e| e.to_string())?;
        demo_log!(
            info,
            KEY,
            "EAS-DEREG eas={} {loc}: {}",
            self.ad.eas_id,
            r.status.as_u16()
        );
        Ok(())
    }
}
