//! A stub eesd for the d443a tests: h2c, the EAS and EEC registration routes
//! the sidecar and the EEC call, with eesd's statuses and `Location`s
//! (`/eees-*/v1/registrations/<id>`), and a scripted EEC selection.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use bytes::{Bytes, BytesMut};
use http::{Response, StatusCode};
use serde_json::{json, Value};
use tokio::net::TcpListener;

use super::{EASREG_PATH, EECREG_PATH, VENDOR_MEMBER};

/// A request the stub saw.
#[derive(Debug, Clone)]
pub struct Seen {
    pub method: String,
    pub path: String,
    pub content_type: Option<String>,
    pub authorization: Option<String>,
    pub body: Vec<u8>,
}

/// How the stub answers an EEC registration.
#[derive(Debug, Clone)]
pub enum EecReply {
    /// 201, `discoveredEas` = these registered EASes in this order, each with
    /// eesd's vendor member `{achievableLatency, budgetSrc}`.
    Select(Vec<(String, u32)>),
    /// This status (404 is "no matching EAS").
    Status(u16),
}

#[derive(Debug)]
struct State {
    eas: BTreeMap<String, Value>,
    eec: BTreeMap<String, Value>,
    next: u32,
    reply: EecReply,
    /// Every request must carry this bearer, or 401 (as eesd's gate).
    bearer: Option<String>,
    seen: Vec<Seen>,
    /// The next PATCHes answer these statuses instead of applying.
    patch_fail: std::collections::VecDeque<u16>,
    /// Answer a registration with an absolute Location on another origin.
    foreign_location: bool,
    /// As an NRF: answer a profile PUT with this heartBeatTimer (else 404).
    nrf_heartbeat: Option<u64>,
}

/// The stub.
#[derive(Debug, Clone)]
pub struct StubEes {
    pub api_root: String,
    state: Arc<Mutex<State>>,
}

impl StubEes {
    pub async fn start() -> Self {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let api_root = format!("http://{}", l.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State {
            eas: BTreeMap::new(),
            eec: BTreeMap::new(),
            next: 0,
            reply: EecReply::Status(404),
            bearer: None,
            seen: Vec::new(),
            patch_fail: std::collections::VecDeque::new(),
            foreign_location: false,
            nrf_heartbeat: None,
        }));
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                let Ok((tcp, _)) = l.accept().await else {
                    continue;
                };
                let st = st.clone();
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(tcp).await else {
                        return;
                    };
                    while let Some(Ok((req, mut respond))) = conn.accept().await {
                        let st = st.clone();
                        tokio::spawn(async move {
                            let (parts, mut body) = req.into_parts();
                            let mut buf = BytesMut::new();
                            while let Some(Ok(c)) = body.data().await {
                                let _ = body.flow_control().release_capacity(c.len());
                                buf.extend_from_slice(&c);
                            }
                            let hdr = |k: http::header::HeaderName| {
                                parts
                                    .headers
                                    .get(k)
                                    .and_then(|v| v.to_str().ok())
                                    .map(str::to_string)
                            };
                            let seen = Seen {
                                method: parts.method.to_string(),
                                path: parts.uri.path().to_string(),
                                content_type: hdr(http::header::CONTENT_TYPE),
                                authorization: hdr(http::header::AUTHORIZATION),
                                body: buf.to_vec(),
                            };
                            let (status, location, out) = route(&st, seen);
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
        Self { api_root, state }
    }

    pub fn set_reply(&self, r: EecReply) {
        self.state.lock().unwrap().reply = r;
    }

    pub fn require_bearer(&self, token: &str) {
        self.state.lock().unwrap().bearer = Some(format!("Bearer {token}"));
    }

    pub fn clear_bearer(&self) {
        self.state.lock().unwrap().bearer = None;
    }

    /// The next PATCHes answer `statuses`, one each, without applying.
    pub fn fail_patches(&self, statuses: &[u16]) {
        self.state.lock().unwrap().patch_fail.extend(statuses);
    }

    /// Act as an NRF that answers a profile PUT with `hb` as heartBeatTimer.
    pub fn nrf_heartbeat(&self, hb: u64) {
        self.state.lock().unwrap().nrf_heartbeat = Some(hb);
    }

    pub fn foreign_location(&self, on: bool) {
        self.state.lock().unwrap().foreign_location = on;
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.state.lock().unwrap().seen.clone()
    }

    /// The stored EAS registrations, by registrationId.
    pub fn eas(&self) -> BTreeMap<String, Value> {
        self.state.lock().unwrap().eas.clone()
    }

    /// The live EEC registrations.
    pub fn eec_count(&self) -> usize {
        self.state.lock().unwrap().eec.len()
    }

    /// Forget every registration (an eesd restart).
    pub fn restart(&self) {
        let mut s = self.state.lock().unwrap();
        s.eas.clear();
        s.eec.clear();
    }
}

fn merge(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(t), Value::Object(p)) => {
            for (k, v) in p {
                if v.is_null() {
                    t.remove(k);
                } else {
                    merge(t.entry(k.clone()).or_insert(Value::Null), v);
                }
            }
        }
        (t, p) => *t = p.clone(),
    }
}

fn route(st: &Mutex<State>, seen: Seen) -> (StatusCode, Option<String>, Option<Value>) {
    let mut s = st.lock().unwrap();
    s.seen.push(seen.clone());
    if let Some(b) = &s.bearer {
        if seen.authorization.as_deref() != Some(b.as_str()) {
            return (StatusCode::UNAUTHORIZED, None, None);
        }
    }
    let v: Value = serde_json::from_slice(&seen.body).unwrap_or(Value::Null);
    let path = seen.path.as_str();
    let id_of = |p: &str, base: &str| p.strip_prefix(base)?.strip_prefix('/').map(str::to_string);
    match seen.method.as_str() {
        "POST" if path == EASREG_PATH => {
            if v["easProf"]["easId"].as_str().is_none() || v["easProf"]["endPt"].is_null() {
                return (StatusCode::BAD_REQUEST, None, None);
            }
            s.next += 1;
            let id = format!("eas-reg-{}", s.next);
            let mut stored = v.clone();
            stored["registrationId"] = json!(id);
            s.eas.insert(id.clone(), stored.clone());
            let loc = if s.foreign_location {
                format!("http://192.0.2.1:7777{EASREG_PATH}/{id}")
            } else {
                format!("{EASREG_PATH}/{id}")
            };
            (StatusCode::CREATED, Some(loc), Some(stored))
        }
        "POST" if path == EECREG_PATH => {
            if v["eecId"].as_str().is_none() {
                return (StatusCode::BAD_REQUEST, None, None);
            }
            match s.reply.clone() {
                EecReply::Status(code) => (StatusCode::from_u16(code).unwrap(), None, None),
                EecReply::Select(order) => {
                    let discovered: Vec<Value> = order
                        .iter()
                        .filter_map(|(eas_id, ms)| {
                            let prof = s
                                .eas
                                .values()
                                .find(|r| r["easProf"]["easId"] == json!(eas_id))?;
                            Some(json!({
                                "eas": prof["easProf"],
                                VENDOR_MEMBER: { "achievableLatency": ms, "budgetSrc": "upf-n3+upf-n6" },
                            }))
                        })
                        .collect();
                    s.next += 1;
                    let id = format!("eec-reg-{}", s.next);
                    let mut out = v.clone();
                    out["registrationId"] = json!(id);
                    if !discovered.is_empty() {
                        out["discoveredEas"] = json!(discovered);
                    }
                    s.eec.insert(id.clone(), out.clone());
                    (
                        StatusCode::CREATED,
                        Some(format!("{EECREG_PATH}/{id}")),
                        Some(out),
                    )
                }
            }
        }
        "PUT" if path.starts_with("/nnrf-nfm/v1/nf-instances/") => match s.nrf_heartbeat {
            Some(hb) => {
                let mut out = v.clone();
                out["heartBeatTimer"] = json!(hb);
                (StatusCode::CREATED, None, Some(out))
            }
            None => (StatusCode::NOT_FOUND, None, None),
        },
        // NF Heart-Beat (TS 29.510 §5.2.2.3.2, #556): 204 for a registered
        // profile, 404 when the stub holds none.
        "PATCH" if path.starts_with("/nnrf-nfm/v1/nf-instances/") => {
            if seen.content_type.as_deref() != Some("application/json-patch+json") {
                return (StatusCode::UNSUPPORTED_MEDIA_TYPE, None, None);
            }
            match s.nrf_heartbeat {
                Some(_) => (StatusCode::NO_CONTENT, None, None),
                None => (StatusCode::NOT_FOUND, None, None),
            }
        }
        "GET" => match id_of(path, EASREG_PATH).and_then(|id| s.eas.get(&id).cloned()) {
            Some(r) => (StatusCode::OK, None, Some(r)),
            None => (StatusCode::NOT_FOUND, None, None),
        },
        "PATCH" => {
            if seen.content_type.as_deref() != Some("application/merge-patch+json") {
                return (StatusCode::UNSUPPORTED_MEDIA_TYPE, None, None);
            }
            if let Some(code) = s.patch_fail.pop_front() {
                return (StatusCode::from_u16(code).unwrap(), None, None);
            }
            let Some(id) = id_of(path, EASREG_PATH) else {
                return (StatusCode::NOT_FOUND, None, None);
            };
            let Some(r) = s.eas.get_mut(&id) else {
                return (StatusCode::NOT_FOUND, None, None);
            };
            merge(r, &v);
            (StatusCode::OK, None, Some(r.clone()))
        }
        "DELETE" => {
            let gone = id_of(path, EASREG_PATH)
                .and_then(|id| s.eas.remove(&id))
                .or_else(|| id_of(path, EECREG_PATH).and_then(|id| s.eec.remove(&id)));
            match gone {
                Some(_) => (StatusCode::NO_CONTENT, None, None),
                None => (StatusCode::NOT_FOUND, None, None),
            }
        }
        _ => (StatusCode::NOT_FOUND, None, None),
    }
}
