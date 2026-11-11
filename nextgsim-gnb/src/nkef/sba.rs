//! NKEF's SBA client (nextgsim#488, feature `nkef-sba`, on the #457 h2c client).
//!
//! Two consumers of nextgcore, both read-only in effect:
//!
//! - **amfd Namf_EventExposure** (TS 29.518 §5.3.2.2): subscribe to
//!   LOCATION_REPORT and CONNECTIVITY_STATE_REPORT for the configured SUPI with
//!   `immediateFlag`, read the answer's `reportList` ("Last Known Location (if the
//!   immediateFlag is set)", `29518-k00.txt:4830-4836`; `:5030` for the CM state),
//!   and unsubscribe at once. **Non-normative:** a RAN node is not a
//!   Namf_EventExposure consumer (the known ones are NEF, SMF, UDM, NWDAF, DCCF,
//!   LMF, TSCTSF and GMLC, `:4814-4815`); NKEF here stands in for an
//!   operator-side exposure function. The notify URI points at a closed loopback
//!   port: NKEF opens no listener and reads only the immediate reports.
//! - **NRF Nnrf_NFDiscovery** (TS 29.510 §5.3.2.2), polled every
//!   `nrf_poll_secs`. The NRF returns only REGISTERED instances, so an NF whose
//!   heartbeat stopped (the NRF suspends it) drops out of the next answer and its
//!   fact expires. The NRF is an observer independent of this gNB.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use nextgsim_common::config::NkefLiveConfig;
use nextgsim_nkef::live::{parse_rfc3339_secs, Domain, Fact, FactKind};
use nextgsim_sbi_client::{SbiClient, SbiClientConfig};
use serde_json::{json, Value};
use tracing::{info, warn};

use super::live::{guami_key, now_ms};
use crate::demo::D447A_KEY;
use crate::tasks::{NkefMessage, TaskHandle};

/// The notify URI amfd is given: a closed loopback port (discard, 9), so no
/// notification can reach anything and NKEF needs no listener.
pub const CLOSED_NOTIFY_URI: &str = "http://127.0.0.1:9/nkef-notify";

/// The whole amfd round (subscribe, unsubscribe) gets at most this long.
pub const AMF_REPORTS_DEADLINE: Duration = Duration::from_secs(3);

/// The requester type NKEF discovers as: it stands in for an operator-side
/// exposure function.
pub const REQUESTER_NF_TYPE: &str = "NEF";

/// NKEF's client toward amfd and the NRF.
#[derive(Debug, Clone)]
pub struct Sba {
    client: Arc<SbiClient>,
    config: NkefLiveConfig,
}

impl Sba {
    /// The client for `config`, or `None` when no AMF SBI URI is set: then
    /// there is no `sba` domain at all.
    pub fn new(config: &NkefLiveConfig) -> Option<Self> {
        config.amf_sbi_uri.as_ref()?;
        Some(Self {
            client: Arc::new(SbiClient::new(SbiClientConfig {
                timeout: Duration::from_secs(2),
                retries: 0,
                ..SbiClientConfig::default()
            })),
            config: config.clone(),
        })
    }

    /// Subscribe with `immediateFlag`, read the `reportList`, unsubscribe, all
    /// within [`AMF_REPORTS_DEADLINE`], so a blackholed amfd cannot stall the
    /// query (and the CLI) past it.
    pub async fn amf_reports(&self) -> Vec<Fact> {
        match tokio::time::timeout(AMF_REPORTS_DEADLINE, self.amf_reports_inner()).await {
            Ok(facts) => facts,
            Err(_) => {
                warn!(
                    "[demo:{D447A_KEY}] NKEF: amfd did not answer within {AMF_REPORTS_DEADLINE:?}"
                );
                Vec::new()
            }
        }
    }

    async fn amf_reports_inner(&self) -> Vec<Fact> {
        let (Some(amf), Some(supi)) = (&self.config.amf_sbi_uri, &self.config.supi) else {
            return Vec::new();
        };
        let amf = amf.trim_end_matches('/');
        let body = subscription_body(supi);
        let resp = match self
            .client
            .post_json(&format!("{amf}/namf-evts/v1/subscriptions"), &body, None)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("[demo:{D447A_KEY}] NKEF: amfd subscribe failed: {e}");
                return Vec::new();
            }
        };
        let status = resp.status.as_u16();
        let answer: Value = resp.json().unwrap_or(Value::Null);
        let location = resp
            .headers
            .get("location")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .or_else(|| {
                answer["subscriptionId"]
                    .as_str()
                    .map(|id| format!("/namf-evts/v1/subscriptions/{id}"))
            });
        let facts = if status == 201 {
            parse_report_list(&answer, supi, now_ms())
        } else {
            Vec::new()
        };
        info!(
            "[demo:{D447A_KEY}] NKEF: amfd subscribe status={status} reports={}",
            facts.len()
        );
        if let Some(loc) = location {
            let uri = if loc.starts_with("http") {
                loc
            } else {
                format!("{amf}{loc}")
            };
            match self.client.delete(&uri, None).await {
                Ok(r) => info!(
                    "[demo:{D447A_KEY}] NKEF: amfd unsubscribe status={}",
                    r.status.as_u16()
                ),
                Err(e) => warn!("[demo:{D447A_KEY}] NKEF: amfd unsubscribe failed: {e}"),
            }
        }
        facts
    }

    /// One discovery round over the configured NF types. `None` when the NRF
    /// could not be asked at all, so an unreachable NRF expires nothing.
    pub async fn discover(&self) -> Option<Vec<Fact>> {
        let nrf = self.config.nrf_uri.as_ref()?.trim_end_matches('/');
        let mut facts = Vec::new();
        for t in &self.config.nf_types {
            let uri = format!(
                "{nrf}/nnrf-disc/v1/nf-instances?target-nf-type={t}&requester-nf-type={REQUESTER_NF_TYPE}"
            );
            match self.client.get_json::<Value>(&uri, None).await {
                Ok(v) => facts.extend(parse_discovery(&v, now_ms())),
                Err(e) => {
                    warn!("[demo:{D447A_KEY}] NKEF: NRF discovery of {t} failed: {e}");
                    return None;
                }
            }
        }
        Some(facts)
    }

    /// Poll the NRF every `nrf_poll_secs` into `nkef`, until its inbox closes.
    pub fn spawn_nrf_poller(&self, nkef: TaskHandle<NkefMessage>) {
        if self.config.nrf_uri.is_none() {
            return;
        }
        let sba = self.clone();
        let period = Duration::from_secs(self.config.nrf_poll_secs.max(1));
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(period);
            loop {
                tick.tick().await;
                if !super::live::tees_on() {
                    continue;
                }
                let Some(facts) = sba.discover().await else {
                    continue;
                };
                let msg = NkefMessage::NrfPoll {
                    facts,
                    at_ms: now_ms(),
                };
                if nkef.send(msg).await.is_err() {
                    break;
                }
            }
        });
    }
}

/// The Namf_EventExposure subscription NKEF sends (TS 29.518 §6.2.6.2.3).
pub fn subscription_body(supi: &str) -> Value {
    json!({
        "subscription": {
            "eventList": [
                { "type": "LOCATION_REPORT", "immediateFlag": true },
                { "type": "CONNECTIVITY_STATE_REPORT", "immediateFlag": true }
            ],
            "eventNotifyUri": CLOSED_NOTIFY_URI,
            "notifyCorrelationId": "nkef-d447a",
            "nfId": "00000000-0000-4000-8000-00000000447a",
            "supi": supi,
        }
    })
}

/// The facts in a subscribe answer's `reportList` (TS 29.518 §6.2.6.2.5
/// AmfEventReport): the last known location and the CM state.
pub fn parse_report_list(answer: &Value, supi: &str, now_ms: u64) -> Vec<Fact> {
    let mut facts = Vec::new();
    for r in answer["reportList"].as_array().into_iter().flatten() {
        match r["type"].as_str() {
            Some("LOCATION_REPORT") => {
                let nr = &r["location"]["nrLocation"];
                if !nr.is_object() {
                    continue;
                }
                let mut f = Fact::new(Domain::Sba, FactKind::UeLocation, "amf", supi, now_ms);
                if let Some(c) = nr["ncgi"]["nrCellId"].as_str() {
                    f = f.with("nrCellId", c);
                }
                if let Some(t) = nr["tai"]["tac"].as_str() {
                    f = f.with("tac", t);
                }
                if let Some(ts) = nr["ueLocationTimestamp"].as_str() {
                    f = f.with("ueLocationTimestamp", ts);
                    if let Some(secs) = parse_rfc3339_secs(ts) {
                        f = f.with("ueLocationTimestampEpoch", secs);
                    }
                }
                facts.push(f);
            }
            Some("CONNECTIVITY_STATE_REPORT") => {
                if let Some(cm) = r["cmInfoList"][0]["cmState"].as_str() {
                    facts.push(
                        Fact::new(Domain::Sba, FactKind::UeConnectivity, "amf", supi, now_ms)
                            .with("cmState", cm),
                    );
                }
            }
            _ => {}
        }
    }
    facts
}

/// The NF instance facts in a discovery answer (TS 29.510 SearchResult).
pub fn parse_discovery(answer: &Value, now_ms: u64) -> Vec<Fact> {
    let mut facts = Vec::new();
    for nf in answer["nfInstances"].as_array().into_iter().flatten() {
        let Some(id) = nf["nfInstanceId"].as_str() else {
            continue;
        };
        let mut f = Fact::new(
            Domain::Sba,
            FactKind::NfInstance,
            "nrf",
            format!("nf-{id}"),
            now_ms,
        )
        .with("nfInstanceId", id)
        .with("nfType", nf["nfType"].as_str().unwrap_or("unknown"))
        .with("nfStatus", nf["nfStatus"].as_str().unwrap_or("unknown"));
        let guamis: Vec<String> = nf["amfInfo"]["guamiList"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|g| {
                Some(guami_key(
                    g["plmnId"]["mcc"].as_str()?,
                    g["plmnId"]["mnc"].as_str()?,
                    g["amfId"].as_str()?,
                ))
            })
            .collect();
        if !guamis.is_empty() {
            f = f.with("guamis", guamis.join(","));
        }
        facts.push(f);
    }
    facts
}

/// The subjects of `facts`, for [`nextgsim_nkef::live::LiveStore::expire_missing`].
pub fn subjects(facts: &[Fact]) -> BTreeSet<String> {
    facts.iter().map(|f| f.subject.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// amfd's answer as its own test reads it back (nextgcore
    /// `positioning_reach_tests.rs`, nextgsim#488's amfd PR).
    #[test]
    fn an_amfd_report_list_becomes_location_and_cm_facts() {
        let answer = json!({
            "subscriptionId": "7",
            "reportList": [
                { "type": "LOCATION_REPORT", "location": { "nrLocation": {
                    "tai": { "plmnId": { "mcc": "001", "mnc": "01" }, "tac": "000001" },
                    "ncgi": { "plmnId": { "mcc": "001", "mnc": "01" }, "nrCellId": "000000010" },
                    "ueLocationTimestamp": "2026-11-11T08:20:27Z" } } },
                { "type": "CONNECTIVITY_STATE_REPORT",
                  "cmInfoList": [{ "cmState": "CONNECTED", "accessType": "3GPP_ACCESS" }] }
            ]
        });
        let facts = parse_report_list(&answer, "imsi-001010000000001", 5);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].attr("nrCellId"), Some("000000010"));
        assert_eq!(
            facts[0].attr("ueLocationTimestampEpoch"),
            Some("1794385227")
        );
        assert_eq!(facts[1].attr("cmState"), Some("CONNECTED"));
        // No reportList (the UE is not known): no fact.
        assert!(parse_report_list(&json!({"subscriptionId": "8"}), "x", 5).is_empty());
    }

    #[test]
    fn a_discovery_answer_becomes_nf_instance_facts_with_guamis() {
        let answer = json!({ "validityPeriod": 3600, "nfInstances": [
            { "nfInstanceId": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "nfType": "AMF",
              "nfStatus": "REGISTERED",
              "amfInfo": { "guamiList": [ { "plmnId": { "mcc": "001", "mnc": "01" },
                                            "amfId": "02556A" } ] } }
        ]});
        let facts = parse_discovery(&answer, 1);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "nf-3fa85f64-5717-4562-b3fc-2c963f66afa6");
        assert_eq!(facts[0].attr("guamis"), Some("001-01-02556a"));
    }

    #[test]
    fn without_an_amf_uri_there_is_no_sba_client() {
        assert!(Sba::new(&NkefLiveConfig::default()).is_none());
        let config = NkefLiveConfig {
            amf_sbi_uri: Some("http://amf:7777".into()),
            ..NkefLiveConfig::default()
        };
        assert!(Sba::new(&config).is_some());
    }

    #[test]
    fn the_subscription_asks_for_immediate_reports_at_a_closed_port() {
        let b = subscription_body("imsi-1");
        assert_eq!(b["subscription"]["eventNotifyUri"], CLOSED_NOTIFY_URI);
        let events = b["subscription"]["eventList"].as_array().unwrap();
        assert!(events.iter().all(|e| e["immediateFlag"] == true));
    }
}
