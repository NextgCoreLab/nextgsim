//! A stub nefd for the AF's tests: h2c, the three routes the AF calls, and a
//! way to deliver a T8 LoC notification to the AF's callback listener over h2c.
//!
//! UE ID retrieval maps a loopback source `127.0.0.N` (N in 10..=99) to the
//! `UeIdInfo` `{"externalId": "ueNNN@fl.nextg"}` (the AF's GPSI
//! `extid-ueNNN@fl.nextg`); any other address is 404, like a BSF with no binding.

use std::sync::{Arc, Mutex};

use bytes::{Bytes, BytesMut};
use http::{Method, Response, StatusCode};
use nextgsim_sbi_client::{SbiClient, SbiClientConfig};
use serde_json::{json, Value};
use tokio::net::TcpListener;

/// One subscription the stub holds.
#[derive(Debug, Clone, PartialEq)]
pub struct StubSub {
    pub id: String,
    pub external_id: String,
    pub destination: String,
    pub deleted: bool,
}

/// The stub.
#[derive(Debug, Clone)]
pub struct StubNef {
    pub api_root: String,
    pub subs: Arc<Mutex<Vec<StubSub>>>,
}

/// The External Identifier the stub gives source address `127.0.0.octet`.
pub fn external_id_of(octet: u8) -> String {
    format!("ue{octet:03}@fl.nextg")
}

/// The AF's GPSI for source address `127.0.0.octet`.
pub fn gpsi_of(octet: u8) -> String {
    format!("extid-{}", external_id_of(octet))
}

impl StubNef {
    /// Serve on a fresh loopback port.
    pub async fn start() -> Self {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let api_root = format!("http://{}", l.local_addr().unwrap());
        let subs = Arc::new(Mutex::new(Vec::new()));
        let s = subs.clone();
        tokio::spawn(async move {
            loop {
                let Ok((tcp, _)) = l.accept().await else {
                    continue;
                };
                let s = s.clone();
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(tcp).await else {
                        return;
                    };
                    while let Some(Ok((req, mut respond))) = conn.accept().await {
                        let s = s.clone();
                        tokio::spawn(async move {
                            let (parts, mut body) = req.into_parts();
                            let mut buf = BytesMut::new();
                            while let Some(Ok(c)) = body.data().await {
                                let _ = body.flow_control().release_capacity(c.len());
                                buf.extend_from_slice(&c);
                            }
                            let (status, location, out) =
                                route(&s, &parts.method, parts.uri.path(), &buf);
                            let mut b = Response::builder().status(status);
                            if let Some(loc) = location {
                                b = b.header(http::header::LOCATION, loc);
                            }
                            let body = out.map(|v| Bytes::from(v.to_string()));
                            let end = body.is_none();
                            let Ok(mut send) = respond.send_response(b.body(()).unwrap(), end)
                            else {
                                return;
                            };
                            if let Some(body) = body {
                                let _ = send.send_data(body, true);
                            }
                        });
                    }
                });
            }
        });
        Self { api_root, subs }
    }

    /// Live (not deleted) subscriptions.
    pub fn live(&self) -> Vec<StubSub> {
        self.subs
            .lock()
            .unwrap()
            .iter()
            .filter(|s| !s.deleted)
            .cloned()
            .collect()
    }

    /// Every subscription ever created.
    pub fn all(&self) -> Vec<StubSub> {
        self.subs.lock().unwrap().clone()
    }

    /// Deliver a LoC report (reason DEREGISTERED) for `gpsi` to the AF, the way
    /// nefd does: an h2c POST of a TS 29.122 MonitoringNotification to the live
    /// subscription's notificationDestination. Returns the AF's status.
    pub async fn notify_loc(&self, gpsi: &str) -> StatusCode {
        let ext = gpsi.strip_prefix("extid-").unwrap();
        let sub = self
            .live()
            .into_iter()
            .rev()
            .find(|s| s.external_id == ext)
            .expect("a live subscription for the UE");
        self.post(&sub.destination, &loc_body(&sub.id, ext)).await
    }

    /// POST `body` to `uri` over h2c.
    pub async fn post(&self, uri: &str, body: &Value) -> StatusCode {
        let c = SbiClient::new(SbiClientConfig::default());
        c.post_json(uri, body, None).await.unwrap().status
    }
}

/// The MonitoringNotification nefd sends for subscription `id`.
pub fn loc_body(id: &str, external_id: &str) -> Value {
    json!({
        "subscription": format!("/3gpp-monitoring-event/v1/fl-af/subscriptions/{id}"),
        "monitoringEventReports": [{
            "monitoringType": "LOSS_OF_CONNECTIVITY",
            "lossOfConnectReason": "DEREGISTERED",
            "externalId": external_id,
        }],
    })
}

fn route(
    subs: &Mutex<Vec<StubSub>>,
    method: &Method,
    path: &str,
    body: &[u8],
) -> (StatusCode, Option<String>, Option<Value>) {
    let v: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    match (method.as_str(), path) {
        ("POST", "/3gpp-ueid/v1/retrieve") => {
            let ip = v
                .pointer("/ueIpAddr/ipv4Addr")
                .and_then(Value::as_str)
                .unwrap_or("");
            let octet = ip
                .strip_prefix("127.0.0.")
                .and_then(|o| o.parse::<u8>().ok())
                .filter(|o| (10..=99).contains(o));
            match octet {
                Some(o) => (
                    StatusCode::OK,
                    None,
                    Some(json!({ "externalId": external_id_of(o) })),
                ),
                None => (
                    StatusCode::NOT_FOUND,
                    None,
                    Some(json!({ "cause": "UE_NOT_FOUND" })),
                ),
            }
        }
        ("POST", "/3gpp-monitoring-event/v1/fl-af/subscriptions") => {
            if v["monitoringType"] != "LOSS_OF_CONNECTIVITY" {
                return (StatusCode::BAD_REQUEST, None, None);
            }
            let mut s = subs.lock().unwrap();
            let id = format!("sub-{}", s.len() + 1);
            s.push(StubSub {
                id: id.clone(),
                external_id: v["externalId"].as_str().unwrap_or("").to_string(),
                destination: v["notificationDestination"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                deleted: false,
            });
            let loc = format!("/3gpp-monitoring-event/v1/fl-af/subscriptions/{id}");
            (StatusCode::CREATED, Some(loc), Some(v))
        }
        ("DELETE", p) => {
            let id = p.rsplit('/').next().unwrap_or("");
            let mut s = subs.lock().unwrap();
            match s.iter_mut().find(|x| x.id == id && !x.deleted) {
                Some(x) => {
                    x.deleted = true;
                    (StatusCode::NO_CONTENT, None, None)
                }
                None => (StatusCode::NOT_FOUND, None, None),
            }
        }
        _ => (StatusCode::NOT_FOUND, None, None),
    }
}
