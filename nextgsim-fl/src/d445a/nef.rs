//! The AF's side of the NEF: UE ID retrieval by source address, T8 monitoring
//! subscribe / delete, and the T8 callback listener.
//!
//! # Calls (h2c, through [`nextgsim_sbi_client`])
//!
//! - **UE ID retrieval** (TS 23.502 §4.15.10): `POST {nef}{ueid_path}` with
//!   `{"afId": <scs_as_id>, "ueIpAddr": {"ipv4Addr": "<source address>"}}`. The
//!   answer's `gpsi` (or `externalId`) is the member's identity. **Provisional
//!   body:** TS 29.522 (Nnef_UEId) is not vendored, so the path and shape are this
//!   demo's contract with nefd (nextgsim#480 plan item 7), and nefd returns the
//!   GPSI in place of the AF-specific External Identifier.
//! - **T8 Monitoring Event** (TS 29.122 §5.3): `POST
//!   {nef}/3gpp-monitoring-event/v1/{scs_as_id}/subscriptions` with
//!   `monitoringType: LOSS_OF_CONNECTIVITY`, the UE's `msisdn` (or `externalId`)
//!   and a `notificationDestination` on the callback listener. The answer's
//!   `Location` is kept, to check notifications against and to DELETE.
//!
//! With `oauth2` configured every call carries an NRF token for an `AF`-typed
//! consumer, or is not sent ([`nextgsim_sbi_client`] fails closed).
//!
//! # The callback listener
//!
//! An h2c listener (nefd's notification client speaks HTTP/2 with prior
//! knowledge). It accepts only `POST {CALLBACK_PREFIX}{id}` for an id this AF
//! minted for one of its own subscriptions: 128 random bits, handed only to the
//! NEF. Anything else is 404 (unknown id), 405 (method) or 400 (body). Bodies are
//! capped at [`MAX_NOTIFICATION`]. It binds 127.0.0.1 by default.

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use http::{Method, Response, StatusCode};
use nextgsim_sbi_client::{CcaKey, OAuth2Config, SbiClient, SbiClientConfig, SbiError, Scope};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, Semaphore};

/// The callback path prefix; the subscription's id follows.
pub const CALLBACK_PREFIX: &str = "/fl-af/v1/t8-notify/";
/// The largest notification body accepted.
pub const MAX_NOTIFICATION: usize = 64 * 1024;
/// Concurrent callback connections accepted.
const MAX_CALLBACK_CONNECTIONS: usize = 32;
/// Concurrent streams answered per callback connection.
const MAX_STREAMS_PER_CONNECTION: usize = 16;
/// The default UE ID retrieval path (provisional; see the module doc).
pub const DEFAULT_UEID_PATH: &str = "/3gpp-ueid/v1/retrieve";

/// The `nef:` block of the AF config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NefConfig {
    /// nefd's API root, e.g. `http://nef:7777`.
    pub api_root: String,
    /// The AF's `scsAsId` (TS 29.122) and `afId`.
    #[serde(default = "default_scs_as_id")]
    pub scs_as_id: String,
    /// The UE ID retrieval path (provisional).
    #[serde(default = "default_ueid_path")]
    pub ueid_path: String,
    /// The token scope of the UE ID call.
    #[serde(default = "default_ueid_scope")]
    pub ueid_scope: String,
    /// The token scope of the monitoring calls.
    #[serde(default = "default_monitoring_scope")]
    pub monitoring_scope: String,
    /// NRF OAuth2 for an `AF`-typed consumer. Absent: calls carry no token, so a
    /// nefd that requires OAuth2 refuses them (401), which the AF reports.
    #[serde(default)]
    pub oauth2: Option<OAuth2Settings>,
    /// Per-call timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

/// `nef.oauth2`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OAuth2Settings {
    /// The NRF's API root.
    pub nrf: String,
    /// This AF's nfInstanceId.
    pub nf_instance_id: String,
    /// Its CCA signing key (created on first use).
    pub cca_key_file: std::path::PathBuf,
    /// Where to publish the public JWK for the NRF, if anywhere.
    #[serde(default)]
    pub public_jwk_dir: Option<std::path::PathBuf>,
    /// Register `nfType: AF` at the NRF at startup (tokenless, as nextgcore's
    /// NRF management API is).
    #[serde(default)]
    pub register: bool,
}

fn default_scs_as_id() -> String {
    "fl-af".to_string()
}
fn default_ueid_path() -> String {
    DEFAULT_UEID_PATH.to_string()
}
fn default_ueid_scope() -> String {
    "nnef-ueid".to_string()
}
fn default_monitoring_scope() -> String {
    "nnef-eventexposure".to_string()
}
fn default_timeout_ms() -> u64 {
    5000
}

impl NefConfig {
    /// Refuse a config that cannot work.
    pub fn validate(&self) -> Result<(), String> {
        if !self.api_root.starts_with("http://") {
            return Err(format!(
                "nef.api_root {:?} must be an http:// (h2c) URI",
                self.api_root
            ));
        }
        if self.scs_as_id.is_empty()
            || !self
                .scs_as_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(format!(
                "nef.scs_as_id {:?} must be [A-Za-z0-9_-]+",
                self.scs_as_id
            ));
        }
        if !self.ueid_path.starts_with('/') {
            return Err(format!(
                "nef.ueid_path {:?} must start with /",
                self.ueid_path
            ));
        }
        Ok(())
    }
}

/// A UE's identity as the NEF gave it: a GPSI (`msisdn-…`, `extid-…`) or an
/// External Identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Gpsi(pub String);

impl std::fmt::Display for Gpsi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Gpsi {
    /// A GPSI from the NEF, checked to be printable and short: it is logged and
    /// sent back to the NEF.
    pub fn parse(s: &str) -> Option<Self> {
        let ok = (1..=64).contains(&s.len())
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.@".contains(&b));
        ok.then(|| Self(s.to_string()))
    }

    /// The TS 29.122 member and value naming this UE in a monitoring request.
    pub fn monitoring_target(&self) -> (&'static str, &str) {
        if let Some(m) = self.0.strip_prefix("msisdn-") {
            ("msisdn", m)
        } else if let Some(e) = self.0.strip_prefix("extid-") {
            ("externalId", e)
        } else {
            ("externalId", &self.0)
        }
    }
}

/// A live T8 subscription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subscription {
    /// The `Location` path the NEF answered (`/3gpp-monitoring-event/v1/…`).
    pub location: String,
    /// The callback id in our `notificationDestination`.
    pub callback_id: String,
}

/// The NEF client.
pub struct Nef {
    cfg: NefConfig,
    client: SbiClient,
}

impl std::fmt::Debug for Nef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nef")
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

fn status_error(what: &str, e: SbiError) -> String {
    match e {
        SbiError::Unauthorized { .. } => format!("{what}: 401 (no valid AF token)"),
        SbiError::Forbidden => format!("{what}: 403 (token not accepted)"),
        other => format!("{what}: {other}"),
    }
}

impl Nef {
    /// A client for `cfg`; with `oauth2`, loads (or creates) the CCA key,
    /// publishes the JWK and registers at the NRF if asked.
    pub async fn new(cfg: NefConfig) -> Result<Self, String> {
        cfg.validate()?;
        let base = SbiClientConfig {
            timeout: Duration::from_millis(cfg.timeout_ms.max(1)),
            ..Default::default()
        };
        let oauth2 = match &cfg.oauth2 {
            None => None,
            Some(o) => {
                let key = CcaKey::load_or_create(&o.cca_key_file).map_err(|e| e.to_string())?;
                if let Some(dir) = &o.public_jwk_dir {
                    key.publish_jwk(dir, &o.nf_instance_id)
                        .map_err(|e| e.to_string())?;
                }
                if o.register {
                    register_af(&o.nrf, &o.nf_instance_id, base.clone()).await?;
                }
                Some(OAuth2Config::new(
                    o.nrf.clone(),
                    o.nf_instance_id.clone(),
                    "AF",
                    key,
                ))
            }
        };
        let client = SbiClient::new(SbiClientConfig { oauth2, ..base });
        Ok(Self { cfg, client })
    }

    fn scope(&self, scope: &str) -> Scope {
        Scope::new("NEF", scope)
    }

    /// The identity of the UE at `ip`.
    pub async fn ue_id(&self, ip: Ipv4Addr) -> Result<Gpsi, String> {
        let uri = format!("{}{}", self.cfg.api_root, self.cfg.ueid_path);
        let body = json!({
            "afId": self.cfg.scs_as_id,
            "ueIpAddr": { "ipv4Addr": ip.to_string() },
        });
        let r = self
            .client
            .post_json(&uri, &body, Some(&self.scope(&self.cfg.ueid_scope)))
            .await
            .map_err(|e| status_error("UE ID retrieval", e))?;
        if !r.status.is_success() {
            return Err(format!(
                "UE ID retrieval for {ip}: status {}",
                r.status.as_u16()
            ));
        }
        let v: Value = r.json().map_err(|e| format!("UE ID retrieval: {e}"))?;
        let id = v
            .get("gpsi")
            .or_else(|| v.get("externalId"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!("UE ID retrieval for {ip}: no gpsi or externalId in the answer")
            })?;
        Gpsi::parse(id).ok_or_else(|| format!("UE ID retrieval for {ip}: unusable identity"))
    }

    /// Subscribe to `LOSS_OF_CONNECTIVITY` for `gpsi`, notifications to
    /// `{notify_base}{CALLBACK_PREFIX}{callback_id}`.
    pub async fn subscribe_loc(
        &self,
        gpsi: &Gpsi,
        notify_base: &str,
        callback_id: &str,
    ) -> Result<Subscription, String> {
        let uri = format!(
            "{}/3gpp-monitoring-event/v1/{}/subscriptions",
            self.cfg.api_root, self.cfg.scs_as_id
        );
        let (member, value) = gpsi.monitoring_target();
        let mut body = json!({
            "monitoringType": "LOSS_OF_CONNECTIVITY",
            "notificationDestination": format!("{notify_base}{CALLBACK_PREFIX}{callback_id}"),
        });
        body[member] = json!(value);
        let r = self
            .client
            .post_json(&uri, &body, Some(&self.scope(&self.cfg.monitoring_scope)))
            .await
            .map_err(|e| status_error("T8 subscribe", e))?;
        if r.status != StatusCode::CREATED {
            return Err(format!(
                "T8 subscribe for {gpsi}: status {}",
                r.status.as_u16()
            ));
        }
        let location = r
            .headers
            .get(http::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .ok_or("T8 subscribe: 201 without a Location")?;
        let location = self.location_path(location).ok_or_else(|| {
            format!("T8 subscribe: Location {location:?} is not under the NEF's monitoring API")
        })?;
        Ok(Subscription {
            location,
            callback_id: callback_id.to_string(),
        })
    }

    /// `Location` as a path under this NEF's monitoring API for our scsAsId, or
    /// `None`: an absolute URI on another authority is never followed.
    fn location_path(&self, location: &str) -> Option<String> {
        let path = match location.strip_prefix(&self.cfg.api_root) {
            Some(p) => p,
            None if location.starts_with('/') => location,
            None => return None,
        };
        let prefix = format!(
            "/3gpp-monitoring-event/v1/{}/subscriptions/",
            self.cfg.scs_as_id
        );
        let id = path.strip_prefix(&prefix)?;
        (!id.is_empty()
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b)))
        .then(|| path.to_string())
    }

    /// Delete a subscription (best effort).
    pub async fn unsubscribe(&self, sub: &Subscription) -> Result<(), String> {
        let uri = format!("{}{}", self.cfg.api_root, sub.location);
        let r = self
            .client
            .delete(&uri, Some(&self.scope(&self.cfg.monitoring_scope)))
            .await
            .map_err(|e| status_error("T8 delete", e))?;
        if r.status.is_success() || r.status == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            Err(format!(
                "T8 delete {}: status {}",
                sub.location,
                r.status.as_u16()
            ))
        }
    }
}

async fn register_af(nrf: &str, id: &str, base: SbiClientConfig) -> Result<(), String> {
    let plain = SbiClient::new(base);
    let profile = json!({
        "nfInstanceId": id,
        "nfType": "AF",
        "nfStatus": "REGISTERED",
        "heartBeatTimer": 60,
    });
    let r = plain
        .put_json(
            &format!("{nrf}/nnrf-nfm/v1/nf-instances/{id}"),
            &profile,
            None,
        )
        .await
        .map_err(|e| format!("NRF register: {e}"))?;
    if !r.status.is_success() {
        return Err(format!("NRF register: status {}", r.status.as_u16()));
    }
    Ok(())
}

/// A fresh callback id: 128 random bits, hex.
pub fn new_callback_id() -> String {
    use rand::RngCore;
    let mut b = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A Loss-of-Connectivity report the NEF delivered.
#[derive(Debug, Clone, PartialEq)]
pub struct LocReport {
    /// The callback id it arrived on.
    pub callback_id: String,
    /// The notification's `subscription` link.
    pub subscription: String,
    /// `lossOfConnectReason`, as sent (a string or an integer, #1444).
    pub reason: Option<Value>,
}

/// Read a TS 29.122 MonitoringNotification: the LoC report in it, or why not.
pub fn parse_loc_notification(callback_id: &str, body: &[u8]) -> Result<LocReport, String> {
    let v: Value = serde_json::from_slice(body).map_err(|e| format!("not JSON: {e}"))?;
    let subscription = v
        .get("subscription")
        .and_then(Value::as_str)
        .ok_or("no subscription link")?
        .to_string();
    let reports = v
        .get("monitoringEventReports")
        .and_then(Value::as_array)
        .ok_or("no monitoringEventReports")?;
    let loc = reports
        .iter()
        .find(|r| r.get("monitoringType").and_then(Value::as_str) == Some("LOSS_OF_CONNECTIVITY"))
        .ok_or("no LOSS_OF_CONNECTIVITY report")?;
    Ok(LocReport {
        callback_id: callback_id.to_string(),
        subscription,
        reason: loc.get("lossOfConnectReason").cloned(),
    })
}

/// The callback ids this AF minted, shared with the listener so an unknown id is
/// answered 404 without reaching the AF's state.
#[derive(Debug, Clone, Default)]
pub struct CallbackIds(Arc<Mutex<HashMap<String, String>>>);

impl CallbackIds {
    /// Register `id` for `label`.
    pub fn insert(&self, id: &str, label: &str) {
        self.lock().insert(id.to_string(), label.to_string());
    }
    /// Forget `id`.
    pub fn remove(&self, id: &str) {
        self.lock().remove(id);
    }
    /// Whether `id` is ours.
    pub fn contains(&self, id: &str) -> bool {
        self.lock().contains_key(id)
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, String>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Serve the T8 callback listener on `listener`, forwarding each accepted body
/// (callback id, bytes) to `tx`.
pub async fn serve_callbacks(
    listener: TcpListener,
    ids: CallbackIds,
    tx: mpsc::Sender<(String, Bytes)>,
) {
    let slots = Arc::new(Semaphore::new(MAX_CALLBACK_CONNECTIONS));
    loop {
        let (tcp, peer) = match listener.accept().await {
            Ok(a) => a,
            Err(e) => {
                // EMFILE and friends: back off rather than spin.
                tracing::warn!("[demo:{}] T8 listener: accept: {e}", super::KEY);
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
        };
        let Ok(permit) = slots.clone().try_acquire_owned() else {
            tracing::warn!(
                "[demo:{}] T8 listener: connection from {peer} refused (busy)",
                super::KEY
            );
            continue;
        };
        let (ids, tx) = (ids.clone(), tx.clone());
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(e) = serve_connection(tcp, ids, tx).await {
                tracing::debug!("[demo:{}] T8 listener: {peer}: {e}", super::KEY);
            }
        });
    }
}

async fn serve_connection(
    tcp: tokio::net::TcpStream,
    ids: CallbackIds,
    tx: mpsc::Sender<(String, Bytes)>,
) -> Result<(), h2::Error> {
    let mut conn =
        match tokio::time::timeout(Duration::from_secs(10), h2::server::handshake(tcp)).await {
            Ok(c) => c?,
            Err(_) => return Ok(()),
        };
    // Each stream is answered in its own task: reading a body needs the
    // connection driven, which only this accept loop does.
    let streams = Arc::new(Semaphore::new(MAX_STREAMS_PER_CONNECTION));
    while let Some(stream) = conn.accept().await {
        let (req, respond) = stream?;
        let Ok(permit) = streams.clone().try_acquire_owned() else {
            let mut respond = respond;
            let resp = Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(())
                .expect("static response");
            let _ = respond.send_response(resp, true);
            continue;
        };
        let (ids, tx) = (ids.clone(), tx.clone());
        tokio::spawn(async move {
            let _permit = permit;
            answer(req, respond, &ids, &tx).await;
        });
    }
    Ok(())
}

async fn answer(
    req: http::Request<h2::RecvStream>,
    mut respond: h2::server::SendResponse<Bytes>,
    ids: &CallbackIds,
    tx: &mpsc::Sender<(String, Bytes)>,
) {
    let (parts, mut body) = req.into_parts();
    let status = match route(&parts.method, parts.uri.path(), ids) {
        Err(s) => s,
        Ok(id) => match read_body(&mut body).await {
            None => StatusCode::BAD_REQUEST,
            Some(bytes) => match parse_loc_notification(&id, &bytes) {
                Err(why) => {
                    tracing::warn!("[demo:{}] T8 callback refused: {why}", super::KEY);
                    StatusCode::BAD_REQUEST
                }
                Ok(_) => {
                    let _ = tx.send((id, bytes)).await;
                    StatusCode::NO_CONTENT
                }
            },
        },
    };
    let resp = Response::builder()
        .status(status)
        .body(())
        .expect("static response");
    let _ = respond.send_response(resp, true);
}

/// The callback id of an acceptable request, or the status refusing it.
fn route(method: &Method, path: &str, ids: &CallbackIds) -> Result<String, StatusCode> {
    let Some(id) = path.strip_prefix(CALLBACK_PREFIX) else {
        return Err(StatusCode::NOT_FOUND);
    };
    if !ids.contains(id) {
        return Err(StatusCode::NOT_FOUND);
    }
    if method != Method::POST {
        return Err(StatusCode::METHOD_NOT_ALLOWED);
    }
    Ok(id.to_string())
}

async fn read_body(body: &mut h2::RecvStream) -> Option<Bytes> {
    let mut out = bytes::BytesMut::new();
    let read = async {
        while let Some(chunk) = body.data().await {
            let chunk = chunk.ok()?;
            let _ = body.flow_control().release_capacity(chunk.len());
            if out.len() + chunk.len() > MAX_NOTIFICATION {
                return None;
            }
            out.extend_from_slice(&chunk);
        }
        Some(())
    };
    tokio::time::timeout(Duration::from_secs(10), read)
        .await
        .ok()??;
    Some(out.freeze())
}

/// Whether `addr` may be advertised to the NEF: never a wildcard.
pub fn advertisable(addr: &SocketAddr) -> bool {
    !addr.ip().is_unspecified()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gpsi_names_its_monitoring_member() {
        let g = Gpsi::parse("msisdn-0010000000003").unwrap();
        assert_eq!(g.monitoring_target(), ("msisdn", "0010000000003"));
        let e = Gpsi::parse("extid-ue3@fl.example").unwrap();
        assert_eq!(e.monitoring_target(), ("externalId", "ue3@fl.example"));
        for bad in ["", "msisdn 1", "a/b", &"x".repeat(65), "ü"] {
            assert_eq!(Gpsi::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_loc_notification_is_read_and_anything_else_refused() {
        let body = br#"{"subscription":"/3gpp-monitoring-event/v1/fl-af/subscriptions/s1",
            "monitoringEventReports":[{"monitoringType":"LOSS_OF_CONNECTIVITY",
            "lossOfConnectReason":"DEREGISTERED","msisdn":"0010000000003"}]}"#;
        let r = parse_loc_notification("cb", body).unwrap();
        assert_eq!(
            r.subscription,
            "/3gpp-monitoring-event/v1/fl-af/subscriptions/s1"
        );
        assert_eq!(r.reason, Some(json!("DEREGISTERED")));
        for bad in [
            &b"nope"[..],
            br#"{"monitoringEventReports":[]}"#,
            br#"{"subscription":"x","monitoringEventReports":[{"monitoringType":"LOCATION_REPORTING"}]}"#,
        ] {
            assert!(parse_loc_notification("cb", bad).is_err());
        }
    }

    #[test]
    fn only_our_callback_ids_are_routed() {
        let ids = CallbackIds::default();
        let id = new_callback_id();
        assert_eq!(id.len(), 32);
        assert_ne!(id, new_callback_id());
        ids.insert(&id, "ue3");
        let path = format!("{CALLBACK_PREFIX}{id}");
        assert_eq!(route(&Method::POST, &path, &ids), Ok(id.clone()));
        assert_eq!(
            route(&Method::GET, &path, &ids),
            Err(StatusCode::METHOD_NOT_ALLOWED)
        );
        assert_eq!(
            route(
                &Method::POST,
                &format!("{CALLBACK_PREFIX}{}", new_callback_id()),
                &ids
            ),
            Err(StatusCode::NOT_FOUND)
        );
        assert_eq!(
            route(&Method::POST, "/other", &ids),
            Err(StatusCode::NOT_FOUND)
        );
        ids.remove(&id);
        assert_eq!(
            route(&Method::POST, &path, &ids),
            Err(StatusCode::NOT_FOUND)
        );
    }

    #[tokio::test]
    async fn a_location_on_another_authority_is_not_followed() {
        let nef = Nef::new(NefConfig {
            api_root: "http://nef:7777".into(),
            scs_as_id: "fl-af".into(),
            ueid_path: DEFAULT_UEID_PATH.into(),
            ueid_scope: default_ueid_scope(),
            monitoring_scope: default_monitoring_scope(),
            oauth2: None,
            timeout_ms: 100,
        })
        .await
        .unwrap();
        let p = "/3gpp-monitoring-event/v1/fl-af/subscriptions/abc-1";
        assert_eq!(nef.location_path(p).as_deref(), Some(p));
        assert_eq!(
            nef.location_path(&format!("http://nef:7777{p}")).as_deref(),
            Some(p)
        );
        assert_eq!(nef.location_path(&format!("http://evil:7777{p}")), None);
        assert_eq!(
            nef.location_path("/3gpp-monitoring-event/v1/other/subscriptions/x"),
            None
        );
        assert_eq!(nef.location_path(&format!("{p}/../../x")), None);
    }
}
