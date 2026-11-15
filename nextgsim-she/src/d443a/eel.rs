//! The Edge Enabler Layer bodies the demo sends, in the shapes the vendored
//! OpenAPI defines, and how an EES selection is read back.
//!
//! - TS 29.558 `EASRegistration` (`TS29558_Eees_EASRegistration.yaml`):
//!   `easProf` (`EASProfile`: `easId`, `endPt`, `svcKpi`, mandatory `easId`
//!   and `endPt`); `EndPoint` is `oneOf` `uri` / `fqdn` / `ipv4Addrs` /
//!   `ipv6Addrs`, and has no port.
//! - `EASRegistrationPatch` (`application/merge-patch+json`): `easProf`, an
//!   `EASProfile`, so the patch repeats `easId` and `endPt` unchanged.
//! - TS 24.558 `EECRegistration` (`TS24558_Eees_EECRegistration.yaml`):
//!   `eecId` (mandatory), `ueId`, `acProfs[]` (`ACProfile`: `acId`, `eass[]`
//!   of `EasDetail` {`easId`, `minimumReqSvcKPIs`}), `easSelReqInd`.
//! - TS 24.558 `DiscoveredEas` (`TS24558_Eees_EASDiscovery.yaml`): `eas` (an
//!   `EASProfile`), plus eesd's vendor-specific member.

use std::net::IpAddr;

use nextgsim_common::config::EecClientConfig;
use serde_json::{json, Value};

use super::VENDOR_MEMBER;

/// The `endPt` of an EAS at `ip`: one address list, nothing else.
pub fn end_point(ip: IpAddr) -> Value {
    match ip {
        IpAddr::V4(v4) => json!({ "ipv4Addrs": [v4.to_string()] }),
        IpAddr::V6(v6) => json!({ "ipv6Addrs": [v6.to_string()] }),
    }
}

/// What an EAS advertises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EasAd {
    pub eas_id: String,
    pub ip: IpAddr,
    /// `svcKpi.maxRespTime`, ms (TS 29.558: includes the AC-EAS round trip
    /// and the EAS processing).
    pub max_resp_time_ms: u32,
    /// `svcKpi.avlComp`.
    pub avl_comp: u32,
    /// `easProf.acIds`, if any.
    pub ac_ids: Vec<String>,
}

/// The `EASRegistration` body of `CreateEASRegistration`.
pub fn eas_registration(ad: &EasAd) -> Value {
    let mut prof = json!({
        "easId": ad.eas_id,
        "endPt": end_point(ad.ip),
        "svcKpi": { "maxRespTime": ad.max_resp_time_ms, "avlComp": ad.avl_comp },
    });
    if !ad.ac_ids.is_empty() {
        prof["acIds"] = json!(ad.ac_ids);
    }
    json!({ "easProf": prof })
}

/// The `EASRegistrationPatch` that changes `svcKpi.avlComp` only.
pub fn avl_comp_patch(ad: &EasAd, avl_comp: u32) -> Value {
    json!({
        "easProf": {
            "easId": ad.eas_id,
            "endPt": end_point(ad.ip),
            "svcKpi": { "avlComp": avl_comp },
        }
    })
}

/// The `EECRegistration` body of `CreateEECReg`: one AC profile naming the
/// configured EASes with the job's `reqComp`, asking the EES to select.
pub fn eec_registration(cfg: &EecClientConfig) -> Value {
    let eass: Vec<Value> = cfg
        .eass
        .iter()
        .map(|id| json!({ "easId": id, "minimumReqSvcKPIs": { "reqComp": cfg.req_comp.to_string() } }))
        .collect();
    let mut body = json!({
        "eecId": cfg.eec_id,
        "acProfs": [{ "acId": cfg.ac_id, "eass": eass }],
        "easSelReqInd": true,
    });
    if let Some(ue_id) = &cfg.ue_id {
        body["ueId"] = json!(ue_id);
    }
    body
}

/// One EAS the EES selected, in the order given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selected {
    pub eas_id: String,
    /// The first address of `eas.endPt`.
    pub ip: Option<IpAddr>,
    /// eesd's labelled `achievableLatency` (ms), if it sent one.
    pub achievable_latency_ms: Option<u32>,
    /// eesd's `budgetSrc`, if it sent one.
    pub budget_src: Option<String>,
}

/// The `discoveredEas` of an `EECRegistration` response, in order. Entries
/// without an `eas.easId` are skipped.
pub fn selection(body: &Value) -> Vec<Selected> {
    body.get("discoveredEas")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|d| {
            let eas = d.get("eas")?;
            let eas_id = eas.get("easId")?.as_str()?.to_string();
            let end_pt = eas.get("endPt");
            let ip = ["ipv4Addrs", "ipv6Addrs"]
                .iter()
                .filter_map(|k| end_pt.and_then(|e| e.get(*k)).and_then(Value::as_array))
                .flatten()
                .find_map(|a| a.as_str().and_then(|s| s.parse::<IpAddr>().ok()));
            let vendor = d.get(VENDOR_MEMBER);
            Some(Selected {
                eas_id,
                ip,
                achievable_latency_ms: vendor
                    .and_then(|v| v.get("achievableLatency"))
                    .and_then(Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok()),
                budget_src: vendor
                    .and_then(|v| v.get("budgetSrc"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect()
}
