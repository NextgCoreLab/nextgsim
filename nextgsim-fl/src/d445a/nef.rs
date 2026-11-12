//! The AF's side of the NEF: UE ID retrieval by source address, T8 monitoring
//! subscribe / delete, and the T8 callback listener.
//!
//! # Calls (h2c, through [`nextgsim_sbi_client`])
//!
//! - **UE ID retrieval** (TS 23.502 §4.15.10; TS 29.522 `3gpp-ueid` API,
//!   RetrieveUEId): `POST {nef}{ueid_path}` with a `UeIdReq`
//!   `{"afId": <scs_as_id>, "ueIpAddr": {"ipv4Addr": "<source address>"}}`. The
//!   answer is a `UeIdInfo`, whose one required member `externalId` is the
//!   AF-specific External Identifier (`username@realm`). The member's identity is
//!   the GPSI `extid-<externalId>` (the TS 29.571 `Gpsi` form), so a member is
//!   logged as `extid-…`. This follows the Rel-18 OpenAPI
//!   (`TS29522_UEId.yaml`, 3gpp-ueid 1.1.1); TS 29.522 is not vendored here, so it
//!   is this demo's contract with nefd (nextgsim#480 plan item 7).
//! - **T8 Monitoring Event** (TS 29.122 §5.3): `POST
//!   {nef}/3gpp-monitoring-event/v1/{scs_as_id}/subscriptions` with
//!   `monitoringType: LOSS_OF_CONNECTIVITY`, the UE's `externalId` (or `msisdn`
//!   for an `msisdn-` GPSI) and a `notificationDestination` on the callback
//!   listener. The answer's `Location` is kept, to check notifications against
//!   and to DELETE.
//!
//! With `oauth2` configured every call carries an NRF token for an `AF`-typed
//! consumer, or is not sent ([`nextgsim_sbi_client`] fails closed). The scopes
//! (`nnef-ueid`, `nnef-eventexposure`) are nextgcore's NRF OAuth2 policy, not 3GPP
//! surface: the OpenAPI defines no scopes for these northbound APIs, which TS
//! 23.222 / TS 33.122 put behind CAPIF authorisation.
//!
//! # The callback listener
//!
//! An h2c listener (nefd's notification client speaks HTTP/2 with prior
//! knowledge). It accepts only `POST {CALLBACK_PREFIX}{id}` for an id this AF
//! minted for one of its own subscriptions: 128 random bits, handed only to the
//! NEF. Anything else is 404 (unknown id), 405 (method) or 400 (body). Bodies are
//! capped at [`MAX_NOTIFICATION`], header lists at 16 KiB, streams at 16 per
//! connection. A connection with nothing in flight is closed after
//! [`CALLBACK_IDLE`]. A source holds at most [`MAX_CONNECTIONS_PER_SOURCE`]
//! connections and past that evicts only its own; at [`MAX_CALLBACK_CONNECTIONS`]
//! eviction takes from the source holding the most, never pushing another source
//! below a fair share; and a connection is not evictable while it opens
//! ([`OPEN_GRACE`]). So one flooding or churning host cannot evict the NEF's
//! connection. It binds 127.0.0.1 by default, and belongs on the network nefd reaches the AF on
//! (SBI or pod network), not on the UEs' `fl-dn`.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http::{Method, Response, StatusCode};
use nextgsim_sbi_client::{CcaKey, OAuth2Config, SbiClient, SbiClientConfig, SbiError, Scope};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, Notify};

/// The callback path prefix; the subscription's id follows.
pub const CALLBACK_PREFIX: &str = "/fl-af/v1/t8-notify/";
/// The largest notification body accepted.
pub const MAX_NOTIFICATION: usize = 64 * 1024;
/// Concurrent callback connections held in all. See [`MAX_CONNECTIONS_PER_SOURCE`]
/// and `Registry::make_room` for who may be evicted when it is reached.
pub const MAX_CALLBACK_CONNECTIONS: usize = 64;
/// Concurrent callback connections held per source address. A source past this
/// evicts its own idle-longest connection, never another source's.
pub const MAX_CONNECTIONS_PER_SOURCE: usize = 8;
/// A new connection cannot be evicted for this long after accept unless it has
/// made its first request: it covers the TCP accept, the h2 handshake and the
/// first HEADERS of a one-notification connection like nefd's.
pub const OPEN_GRACE: Duration = Duration::from_secs(2);
/// Concurrent streams per callback connection, advertised in SETTINGS.
const MAX_STREAMS_PER_CONNECTION: usize = 16;
/// The largest header list accepted (SETTINGS_MAX_HEADER_LIST_SIZE); h2's own
/// default is 16 MiB.
const MAX_HEADER_LIST: u32 = 16 * 1024;
/// A draining connection accepts (and answers) new streams for this long from
/// the start of its drain: long enough for a request that raced the eviction.
const EVICT_DRAIN: Duration = Duration::from_millis(250);
/// A draining connection is closed this long after its drain started,
/// whatever is still in flight. Never extended.
const DRAIN_BUSY: Duration = Duration::from_secs(2);
/// Draining connections held at once. Past this, the one draining longest is
/// closed at once. With the per-source and total caps this bounds the
/// listener's sockets at [`MAX_CALLBACK_CONNECTIONS`] live plus this many
/// draining, each for at most [`DRAIN_BUSY`].
pub const MAX_DRAINING: usize = 64;
/// At most one warn summary of refused callback connections per this long.
const REFUSAL_SUMMARY: Duration = Duration::from_secs(10);
/// A callback connection with no request in flight for this long is closed.
pub const CALLBACK_IDLE: Duration = Duration::from_secs(30);
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

/// The member identity in a TS 29.522 `UeIdInfo`: the GPSI `extid-<externalId>`.
/// `externalId` is the schema's one required member; anything else (a `gpsi`
/// member, say, which the schema does not define) is not read.
pub fn gpsi_of_ue_id_info(v: &Value) -> Result<Gpsi, String> {
    let ext = v
        .get("externalId")
        .and_then(Value::as_str)
        .ok_or("no externalId in the UeIdInfo")?;
    if !ext.contains('@') {
        return Err(format!("externalId {ext:?} is not username@realm"));
    }
    Gpsi::parse(&format!("extid-{ext}")).ok_or_else(|| format!("unusable externalId {ext:?}"))
}

/// A UE's identity: a GPSI (`extid-…` from the NEF's UE ID retrieval, or
/// `msisdn-…`).
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
        gpsi_of_ue_id_info(&v).map_err(|why| format!("UE ID retrieval for {ip}: {why}"))
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

/// One held callback connection, as the listener's registry sees it.
struct Held {
    id: u64,
    /// The peer's address: eviction stays within a source.
    source: IpAddr,
    /// Requests in flight on it.
    busy: Arc<AtomicUsize>,
    /// Still opening: from TCP accept through the h2 handshake to its first
    /// request, or [`OPEN_GRACE`]. Its own source cannot evict it then.
    opening: Arc<AtomicBool>,
    /// When it last had a request in flight (or was accepted).
    last_active: Arc<Mutex<Instant>>,
    /// Told to close (evicted).
    evict: Arc<Notify>,
    /// When it started draining (evicted, or closing after [`CALLBACK_IDLE`]),
    /// shared with its task. A draining connection is out of the admission
    /// counts, is never picked as a victim, and lasts at most [`DRAIN_BUSY`];
    /// draining connections are capped on their own ([`MAX_DRAINING`]).
    draining: Arc<Mutex<Option<Instant>>>,
    /// Told to close at once because the draining cap was reached.
    killed: bool,
}

impl Held {
    fn is_draining(&self) -> bool {
        lock(&self.draining).is_some()
    }
    fn idle(&self) -> bool {
        self.busy.load(Ordering::SeqCst) == 0
    }
    fn opening(&self) -> bool {
        self.opening.load(Ordering::SeqCst)
    }
}

/// The listener's held connections.
#[derive(Default)]
struct Registry {
    next: u64,
    held: Vec<Held>,
}

impl Registry {
    /// Connections not draining: the ones admission counts.
    fn live(&self) -> impl Iterator<Item = &Held> {
        self.held.iter().filter(|h| !h.is_draining())
    }

    fn count(&self, source: IpAddr) -> usize {
        self.live().filter(|h| h.source == source).count()
    }

    /// Draining connections not yet told to close at once.
    fn draining(&self) -> usize {
        self.held
            .iter()
            .filter(|h| !h.killed && h.is_draining())
            .count()
    }

    /// Past [`MAX_DRAINING`], close the connection draining longest at once.
    fn cap_draining(&mut self) {
        while self.draining() > MAX_DRAINING {
            let oldest = self
                .held
                .iter()
                .enumerate()
                .filter(|(_, h)| !h.killed)
                .filter_map(|(i, h)| lock(&h.draining).map(|t| (i, t)))
                .min_by_key(|(_, t)| *t)
                .map(|(i, _)| i);
            let Some(i) = oldest else { return };
            self.held[i].killed = true;
            self.held[i].evict.notify_one();
        }
    }

    /// Evict `source`'s connection idle longest, skipping one still opening
    /// unless `even_opening`; `false` if there is none to evict.
    fn evict_oldest_idle_of(&mut self, source: IpAddr, even_opening: bool) -> bool {
        let oldest = self
            .held
            .iter()
            .enumerate()
            .filter(|(_, h)| {
                h.source == source && !h.is_draining() && h.idle() && (even_opening || !h.opening())
            })
            .min_by_key(|(_, h)| *lock(&h.last_active))
            .map(|(i, _)| i);
        match oldest {
            Some(i) => {
                // Out of the admission counts from now; draining is bounded by
                // its own cap and deadline.
                *lock(&self.held[i].draining) = Some(Instant::now());
                self.held[i].evict.notify_one();
                self.cap_draining();
                true
            }
            None => false,
        }
    }

    /// Make room for one more connection from `source`. `false`: refuse it.
    ///
    /// - A connection with a request in flight is never evicted.
    /// - A source holds at most [`MAX_CONNECTIONS_PER_SOURCE`]; past that its
    ///   own idle-longest connection that has finished opening is evicted, never
    ///   another source's (and the new one is refused if there is none).
    /// - At [`MAX_CALLBACK_CONNECTIONS`] in all, the evicted connection comes from
    ///   the source holding the most, opening or not, and only if that source
    ///   holds at least two more than `source` does. A source can be pushed
    ///   down to a fair share, never by a source holding fewer: a peer with one
    ///   connection (nefd's one notification) is never evicted by a newcomer.
    fn make_room(&mut self, source: IpAddr) -> bool {
        let mine = self.count(source);
        if mine >= MAX_CONNECTIONS_PER_SOURCE {
            return self.evict_oldest_idle_of(source, false);
        }
        if self.live().count() < MAX_CALLBACK_CONNECTIONS {
            return true;
        }
        let mut by_source: Vec<(IpAddr, usize)> = Vec::new();
        for h in self.live() {
            match by_source.iter_mut().find(|(s, _)| *s == h.source) {
                Some((_, n)) => *n += 1,
                None => by_source.push((h.source, 1)),
            }
        }
        by_source.sort_by_key(|s| std::cmp::Reverse(s.1));
        by_source
            .into_iter()
            .take_while(|(_, n)| *n > mine + 1)
            .any(|(s, _)| self.evict_oldest_idle_of(s, true))
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Serve the T8 callback listener on `listener`, forwarding each accepted body
/// (callback id, bytes) to `tx`.
pub async fn serve_callbacks(
    listener: TcpListener,
    ids: CallbackIds,
    tx: mpsc::Sender<(String, Bytes)>,
) {
    serve_callbacks_with(listener, ids, tx, REFUSAL_SUMMARY).await;
}

/// Refused callback connections, summarised at most once per `every`.
struct Refusals {
    every: Duration,
    count: u64,
    latest: Option<SocketAddr>,
    due: tokio::time::Instant,
}

impl Refusals {
    fn new(every: Duration) -> Self {
        Self {
            every,
            count: 0,
            latest: None,
            due: tokio::time::Instant::now(),
        }
    }

    fn refused(&mut self, peer: SocketAddr) {
        self.count += 1;
        self.latest = Some(peer);
        tracing::debug!(
            "[demo:{}] T8 listener: connection from {peer} refused (no idle connection to \
             evict for it)",
            super::KEY
        );
        if tokio::time::Instant::now() >= self.due {
            self.flush();
        }
    }

    /// Warn with the window's count, if any, and start the next window.
    fn flush(&mut self) {
        if self.count > 0 {
            tracing::warn!(
                "[demo:{}] T8 listener: {} connection(s) refused since the last summary \
                 (latest from {}): every slot its source may take is busy or opening",
                super::KEY,
                self.count,
                self.latest
                    .map_or_else(|| "?".to_string(), |p| p.to_string())
            );
        }
        self.count = 0;
        self.due = tokio::time::Instant::now() + self.every;
    }
}

async fn serve_callbacks_with(
    listener: TcpListener,
    ids: CallbackIds,
    tx: mpsc::Sender<(String, Bytes)>,
    summary_every: Duration,
) {
    let registry = Arc::new(Mutex::new(Registry::default()));
    // Refusals are wire-driven: each is logged at debug, and a warn summary
    // goes out at most once per `summary_every`, flushed on a timer so the
    // last window of a burst is reported too.
    let mut refusals = Refusals::new(summary_every);
    loop {
        let accepted = tokio::select! {
            a = listener.accept() => a,
            () = tokio::time::sleep_until(refusals.due), if refusals.count > 0 => {
                refusals.flush();
                continue;
            }
        };
        let (tcp, peer) = match accepted {
            Ok(a) => a,
            Err(e) => {
                // EMFILE and friends: back off rather than spin.
                tracing::warn!("[demo:{}] T8 listener: accept: {e}", super::KEY);
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
        };
        let held = {
            let mut r = lock(&registry);
            if !r.make_room(peer.ip()) {
                drop(r);
                refusals.refused(peer);
                continue;
            }
            r.next += 1;
            let h = Held {
                id: r.next,
                source: peer.ip(),
                busy: Arc::new(AtomicUsize::new(0)),
                // Opening from accept.
                opening: Arc::new(AtomicBool::new(true)),
                last_active: Arc::new(Mutex::new(Instant::now())),
                evict: Arc::new(Notify::new()),
                draining: Arc::new(Mutex::new(None)),
                killed: false,
            };
            let mine = (
                h.id,
                h.busy.clone(),
                h.opening.clone(),
                h.last_active.clone(),
                h.evict.clone(),
                h.draining.clone(),
            );
            r.held.push(h);
            mine
        };
        let (ids, tx, registry) = (ids.clone(), tx.clone(), registry.clone());
        tokio::spawn(async move {
            let (id, busy, opening, last_active, evict, draining) = held;
            let conn = ConnState {
                busy,
                opening,
                last_active,
                evict,
                draining,
            };
            if let Err(e) = serve_connection(tcp, ids, tx, &conn).await {
                tracing::debug!("[demo:{}] T8 listener: {peer}: {e}", super::KEY);
            }
            lock(&registry).held.retain(|h| h.id != id);
        });
    }
}

/// Decrements a connection's in-flight count when a request is answered.
struct InFlight(Arc<AtomicUsize>, Arc<Mutex<Instant>>);

impl Drop for InFlight {
    fn drop(&mut self) {
        *lock(&self.1) = Instant::now();
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A held connection's shared state (see [`Held`]).
struct ConnState {
    busy: Arc<AtomicUsize>,
    opening: Arc<AtomicBool>,
    last_active: Arc<Mutex<Instant>>,
    evict: Arc<Notify>,
    draining: Arc<Mutex<Option<Instant>>>,
}

async fn serve_connection(
    tcp: tokio::net::TcpStream,
    ids: CallbackIds,
    tx: mpsc::Sender<(String, Bytes)>,
    st: &ConnState,
) -> Result<(), h2::Error> {
    let ConnState {
        busy,
        last_active,
        evict,
        ..
    } = st;
    // The connection is "opening" until its first request or OPEN_GRACE after
    // accept, whichever comes first.
    let open_until = tokio::time::Instant::now() + OPEN_GRACE;
    let mut opening = true;
    let end_opening = |opening: &mut bool| {
        if std::mem::take(opening) {
            *lock(last_active) = Instant::now();
            st.opening.store(false, Ordering::SeqCst);
        }
    };
    let handshake = h2::server::Builder::new()
        .max_concurrent_streams(MAX_STREAMS_PER_CONNECTION as u32)
        .max_header_list_size(MAX_HEADER_LIST)
        .handshake(tcp);
    let handshake = tokio::time::timeout(Duration::from_secs(10), handshake);
    tokio::pin!(handshake);
    let mut conn = loop {
        tokio::select! {
            c = &mut handshake => match c {
                Ok(c) => break c?,
                Err(_) => return Ok(()),
            },
            () = tokio::time::sleep_until(open_until), if opening => end_opening(&mut opening),
            () = evict.notified() => return Ok(()),
        }
    };
    // Each stream is answered in its own task: reading a body needs the
    // connection driven, which only this accept loop does.
    loop {
        let idle_left = if busy.load(Ordering::SeqCst) == 0 {
            CALLBACK_IDLE.saturating_sub(lock(last_active).elapsed())
        } else {
            CALLBACK_IDLE
        };
        let stream = tokio::select! {
            s = conn.accept() => s,
            () = tokio::time::sleep_until(open_until), if opening => {
                end_opening(&mut opening);
                continue;
            }
            () = tokio::time::sleep(idle_left) => {
                if busy.load(Ordering::SeqCst) == 0
                    && lock(last_active).elapsed() >= CALLBACK_IDLE
                {
                    tracing::debug!("[demo:{}] T8 listener: idle connection closed", super::KEY);
                    // Mark it draining first, so the registry never picks it.
                    lock(&st.draining).get_or_insert_with(Instant::now);
                    break;
                }
                continue;
            }
            () = evict.notified() => {
                // Idle as the registry last saw it, but a request may already be
                // on the wire: drain (GOAWAY, then answer what the peer had
                // sent), within the drain's absolute deadlines.
                tracing::debug!("[demo:{}] T8 listener: idle connection evicted", super::KEY);
                break;
            }
        };
        let Some(stream) = stream else {
            return Ok(());
        };
        let (req, respond) = stream?;
        busy.fetch_add(1, Ordering::SeqCst);
        *lock(last_active) = Instant::now();
        end_opening(&mut opening);
        let guard = InFlight(busy.clone(), last_active.clone());
        let (ids, tx) = (ids.clone(), tx.clone());
        tokio::spawn(async move {
            let _guard = guard;
            answer(req, respond, &ids, &tx).await;
        });
    }
    // GOAWAY, then answer the streams the peer opened before it. Absolute
    // deadlines from the start of the drain, never extended: streams are
    // accepted (and answered) until EVICT_DRAIN, later ones are refused, and
    // at DRAIN_BUSY the connection is closed whatever is still in flight. A
    // second eviction notice (the draining cap) closes it at once.
    conn.graceful_shutdown();
    let now = tokio::time::Instant::now;
    let start = now();
    let accept_until = start + EVICT_DRAIN;
    let hard = start + DRAIN_BUSY;
    loop {
        let accepting = now() < accept_until;
        if !accepting && busy.load(Ordering::SeqCst) == 0 {
            break;
        }
        let stream = tokio::select! {
            s = conn.accept() => s,
            () = tokio::time::sleep_until(accept_until), if accepting => continue,
            () = tokio::time::sleep(Duration::from_millis(20)), if !accepting => continue,
            () = tokio::time::sleep_until(hard) => break,
            () = evict.notified() => break,
        };
        let Some(Ok((req, mut respond))) = stream else {
            break;
        };
        if now() >= accept_until {
            respond.send_reset(h2::Reason::REFUSED_STREAM);
            continue;
        }
        busy.fetch_add(1, Ordering::SeqCst);
        *lock(last_active) = Instant::now();
        let guard = InFlight(busy.clone(), last_active.clone());
        let (ids, tx) = (ids.clone(), tx.clone());
        tokio::spawn(async move {
            let _guard = guard;
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
    use tokio::io::AsyncReadExt;

    #[test]
    fn a_gpsi_names_its_monitoring_member() {
        let id = gpsi_of_ue_id_info(&json!({ "externalId": "ue013@fl.nextg" })).unwrap();
        assert_eq!(id.0, "extid-ue013@fl.nextg");
        assert_eq!(id.monitoring_target(), ("externalId", "ue013@fl.nextg"));
        for bad in [
            json!({ "gpsi": "msisdn-0010000000003" }),
            json!({ "externalId": "no-realm" }),
            json!({ "externalId": "a b@c" }),
            json!({ "externalId": 7 }),
        ] {
            assert!(gpsi_of_ue_id_info(&bad).is_err(), "{bad}");
        }
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
            "lossOfConnectReason":"DEREGISTERED","externalId":"ue003@fl.nextg"}]}"#;
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

    fn held(r: &mut Registry, octet: u8, opening: bool) -> Arc<Notify> {
        held_busy(r, octet, opening, 0)
    }

    fn held_busy(r: &mut Registry, octet: u8, opening: bool, busy: usize) -> Arc<Notify> {
        r.next += 1;
        let evict = Arc::new(Notify::new());
        r.held.push(Held {
            id: r.next,
            source: IpAddr::from([127, 0, 0, octet]),
            busy: Arc::new(AtomicUsize::new(busy)),
            opening: Arc::new(AtomicBool::new(opening)),
            last_active: Arc::new(Mutex::new(Instant::now())),
            evict: evict.clone(),
            draining: Arc::new(Mutex::new(None)),
            killed: false,
        });
        std::thread::sleep(Duration::from_millis(1));
        evict
    }

    async fn evicted(n: &Notify) -> bool {
        tokio::time::timeout(Duration::from_millis(5), n.notified())
            .await
            .is_ok()
    }

    /// The eviction rules, deterministically. nefd's one connection is the
    /// oldest idle one in a full listener, yet a newcomer never evicts it: at
    /// the total cap the victim comes from the source holding the most, and
    /// a source at its own cap evicts only its own, never one still opening.
    #[tokio::test]
    async fn eviction_takes_from_the_largest_source_and_never_the_nefs_one() {
        let mut r = Registry::default();
        let nef = held(&mut r, 2, false);
        let flood: Vec<_> = (0..MAX_CALLBACK_CONNECTIONS - 1)
            .map(|i| held(&mut r, 20 + (i % 9) as u8, false))
            .collect();
        assert_eq!(r.held.len(), MAX_CALLBACK_CONNECTIONS);
        // A fresh source at the total cap: room is made, not from nefd.
        assert!(r.make_room(IpAddr::from([127, 0, 0, 50])));
        assert!(!evicted(&nef).await, "nefd's connection was evicted");
        let mut gone = 0;
        for n in &flood {
            gone += usize::from(evicted(n).await);
        }
        assert_eq!(gone, 1);
        // A source at its own cap evicts its own oldest, and refuses the new
        // connection when all of its own are still opening.
        let mut r = Registry::default();
        let other = held(&mut r, 2, false);
        let own: Vec<_> = (0..MAX_CONNECTIONS_PER_SOURCE)
            .map(|_| held(&mut r, 30, false))
            .collect();
        assert!(r.make_room(IpAddr::from([127, 0, 0, 30])));
        assert!(evicted(&own[0]).await && !evicted(&other).await);
        let mut r = Registry::default();
        let _opening: Vec<_> = (0..MAX_CONNECTIONS_PER_SOURCE)
            .map(|_| held(&mut r, 30, true))
            .collect();
        assert!(!r.make_room(IpAddr::from([127, 0, 0, 30])));
    }

    fn src(octet: u8) -> IpAddr {
        IpAddr::from([127, 0, 0, octet])
    }

    /// Boundary: at the total cap, a source holding exactly one more than the
    /// newcomer's is not evicted (that would only swap their counts and
    /// ping-pong); the newcomer is refused.
    #[tokio::test]
    async fn no_eviction_when_the_largest_holds_only_one_more() {
        let mut r = Registry::default();
        // The newcomer's source (.2) holds 1; the other 63 slots go to 31
        // sources holding 2 each and one holding 1.
        let mine = held(&mut r, 2, false);
        let others: Vec<_> = (0..MAX_CALLBACK_CONNECTIONS - 1)
            .map(|i| held(&mut r, 20 + (i / 2) as u8, false))
            .collect();
        assert_eq!(r.held.len(), MAX_CALLBACK_CONNECTIONS);
        // Newcomer from source .2 (holds 1); the largest hold 2 == 1 + 1.
        assert!(!r.make_room(src(2)), "evicted at n == mine + 1");
        assert!(!evicted(&mine).await);
        for n in &others {
            assert!(!evicted(n).await, "evicted at n == mine + 1");
        }
    }

    /// Boundary: a connection with a request in flight is never evicted,
    /// neither by its own source at the per-source cap nor by another source
    /// at the total cap.
    #[tokio::test]
    async fn a_busy_connection_is_never_evicted() {
        let mut r = Registry::default();
        let own: Vec<_> = (0..MAX_CONNECTIONS_PER_SOURCE)
            .map(|_| held_busy(&mut r, 30, false, 1))
            .collect();
        assert!(!r.make_room(src(30)), "a busy own connection was evicted");
        let mut r = Registry::default();
        let busy: Vec<_> = (0..MAX_CALLBACK_CONNECTIONS)
            .map(|i| held_busy(&mut r, 20 + (i % 8) as u8, false, 1))
            .collect();
        assert!(!r.make_room(src(50)), "a busy connection was evicted");
        for n in own.iter().chain(&busy) {
            assert!(!evicted(n).await);
        }
    }

    /// Boundary: at the total cap, another source's newcomer CAN evict a
    /// connection that is still opening (the opening window protects a
    /// connection only from its own source), so opening slots cannot be held
    /// against other sources.
    #[tokio::test]
    async fn an_opening_connection_is_evicted_at_the_total_cap() {
        let mut r = Registry::default();
        let opening: Vec<_> = (0..MAX_CALLBACK_CONNECTIONS)
            .map(|i| held(&mut r, 20 + (i % 8) as u8, true))
            .collect();
        assert!(
            r.make_room(src(50)),
            "opening slots held against another source"
        );
        let mut gone = 0;
        for n in &opening {
            gone += usize::from(evicted(n).await);
        }
        assert_eq!(gone, 1);
    }

    /// #533 MAJOR-1: admit-then-push over and over from one source, the way
    /// the accept loop does. Draining connections leave the admission counts
    /// (so they never cause a refusal), the source never holds more than its
    /// cap live, draining is capped on its own with the longest-draining one
    /// told to close at once, and nefd is still admitted. A connection
    /// closing after its idle timeout (m-2) is draining too, never a victim.
    #[tokio::test]
    async fn drains_leave_admission_and_are_capped_on_their_own() {
        let mut r = Registry::default();
        let mut killed = 0;
        for _ in 0..(MAX_CONNECTIONS_PER_SOURCE + MAX_DRAINING) * 3 {
            assert!(r.make_room(src(50)), "the flooder refused its own churn");
            held(&mut r, 50, false);
            // A killed connection's task ends at once and leaves the registry.
            killed += r.held.iter().filter(|h| h.killed).count();
            r.held.retain(|h| !h.killed);
            let live = r.held.iter().filter(|h| !h.is_draining()).count();
            assert!(live <= MAX_CONNECTIONS_PER_SOURCE, "{live} live");
            assert!(r.draining() <= MAX_DRAINING, "{} draining", r.draining());
        }
        assert!(killed > 0, "the draining cap never closed anything");
        assert!(
            r.held.len() <= MAX_CONNECTIONS_PER_SOURCE + MAX_DRAINING,
            "{} sockets held",
            r.held.len()
        );
        assert!(r.make_room(src(2)), "nefd refused");
        // m-2: an idle-closing connection is draining, so it is never picked.
        let mut r = Registry::default();
        let closing = held(&mut r, 30, false);
        *lock(&r.held[0].draining) = Some(Instant::now());
        let own: Vec<_> = (1..MAX_CONNECTIONS_PER_SOURCE)
            .map(|_| held(&mut r, 30, false))
            .collect();
        assert_eq!(r.count(src(30)), MAX_CONNECTIONS_PER_SOURCE - 1);
        held(&mut r, 30, false);
        assert!(r.make_room(src(30)));
        assert!(
            !evicted(&closing).await,
            "an idle-closing connection was picked"
        );
        assert!(evicted(&own[0]).await);
    }

    /// Raw HTTP/2 from a peer that never answers PING (so a graceful
    /// shutdown never completes on its side): the preface, an empty SETTINGS,
    /// and `GET /` on stream `id`.
    fn h2_preface() -> Vec<u8> {
        let mut b = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
        b.extend_from_slice(&[0, 0, 0, 4, 0, 0, 0, 0, 0]);
        b
    }

    fn h2_get(id: u32) -> Vec<u8> {
        // :method GET, :scheme http, :path /, :authority "a" (not indexed).
        let block = [0x82, 0x86, 0x84, 0x01, 0x01, b'a'];
        let mut b = vec![0, 0, block.len() as u8, 1, 0x05];
        b.extend_from_slice(&(id & 0x7fff_ffff).to_be_bytes());
        b.extend_from_slice(&block);
        b
    }

    /// A never-PING peer that keeps sending `GET /` every `every` until the
    /// connection fails; reads and discards everything (no PONG, no ACK).
    async fn never_pong(mut tcp: tokio::net::TcpStream, every: Duration) {
        use tokio::io::AsyncWriteExt;
        if tcp.write_all(&h2_preface()).await.is_err() {
            return;
        }
        let (mut rd, mut wr) = tcp.into_split();
        let reader = async move {
            let mut buf = [0u8; 4096];
            while !matches!(rd.read(&mut buf).await, Ok(0) | Err(_)) {}
        };
        let writer = async move {
            let mut id = 1u32;
            loop {
                if wr.write_all(&h2_get(id)).await.is_err() {
                    return;
                }
                id += 2;
                tokio::time::sleep(every).await;
            }
        };
        tokio::select! {
            () = reader => {}
            () = writer => {}
        }
    }

    /// #533 MAJOR-1: a draining connection is closed at its absolute deadline
    /// even when the peer never answers the GOAWAY's PING and keeps opening
    /// streams: no accepted stream extends the drain.
    #[test]
    fn a_streaming_drain_ends_at_its_deadline() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = l.local_addr().unwrap();
            let (tx, _rx) = mpsc::channel(64);
            let st = ConnState {
                busy: Arc::new(AtomicUsize::new(0)),
                opening: Arc::new(AtomicBool::new(true)),
                last_active: Arc::new(Mutex::new(Instant::now())),
                evict: Arc::new(Notify::new()),
                draining: Arc::new(Mutex::new(None)),
            };
            let evict = st.evict.clone();
            let peer = tokio::spawn(async move {
                let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
                never_pong(tcp, Duration::from_millis(50)).await;
            });
            let (tcp, _) = l.accept().await.unwrap();
            let server = tokio::spawn(async move {
                serve_connection(tcp, CallbackIds::default(), tx, &st).await
            });
            tokio::time::sleep(Duration::from_millis(300)).await;
            let evicted_at = tokio::time::Instant::now();
            evict.notify_one();
            tokio::time::timeout(DRAIN_BUSY + Duration::from_secs(1), server)
                .await
                .expect("the drain outlived its deadline")
                .unwrap()
                .ok();
            assert!(evicted_at.elapsed() <= DRAIN_BUSY + Duration::from_millis(500));
            peer.abort();
        });
    }

    fn open_fds() -> usize {
        std::fs::read_dir("/proc/self/fd").map_or(0, Iterator::count)
    }

    /// #533 MAJOR-1, through the real accept loop: 300 connections from one
    /// source that never answer PING and keep sending `GET /` churn far past
    /// the caps. The process's open sockets stay bounded (the listener holds
    /// at most 8 live from that source plus [`MAX_DRAINING`] draining, each
    /// for at most [`DRAIN_BUSY`]), and nefd's notifications from another
    /// source all get 204. Ignored in the default suite (it floods loopback):
    /// run serially with the churn smoke tests.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "smoke test; run serially with --ignored --test-threads=1"]
    async fn a_churning_never_pong_flood_keeps_sockets_bounded_and_nefd_admitted() {
        let (addr, id, mut rx) = listener().await;
        let base = open_fds();
        let stop = Arc::new(AtomicBool::new(false));
        let flooding = stop.clone();
        let flood = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                for i in 0..300u64 {
                    if let Ok(tcp) = tcp_from(50, addr).await {
                        tokio::spawn(never_pong(tcp, Duration::from_millis(15 + i % 86)));
                    }
                    tokio::time::sleep(Duration::from_millis(3)).await;
                }
                while !flooding.load(Ordering::Relaxed) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            });
            rt.shutdown_background();
        });
        let mut peak = 0;
        let mut statuses = Vec::new();
        for _ in 0..12 {
            tokio::time::sleep(Duration::from_millis(250)).await;
            peak = peak.max(open_fds().saturating_sub(base));
            statuses.push(notify_from(2, addr, &id).await);
        }
        stop.store(true, Ordering::Relaxed);
        flood.join().unwrap();
        // Both ends of every socket are in this process: at most 2 x (8 live
        // + 64 draining + nefd's), plus slack for the runtimes.
        let bound = 2 * (MAX_CONNECTIONS_PER_SOURCE + MAX_DRAINING + 2) + 40;
        assert!(peak <= bound, "{peak} fds open (bound {bound})");
        assert!(
            statuses.iter().all(|s| *s == Ok(StatusCode::NO_CONTENT)),
            "nefd refused: {statuses:?}"
        );
        let mut forwarded = 0;
        while rx.try_recv().is_ok() {
            forwarded += 1;
        }
        assert_eq!(forwarded, 12);
    }

    /// MINOR-1 (#531 round 4): a request already on the wire when its
    /// connection is evicted (idle as the registry saw it, `busy == 0`) is
    /// answered, not cut. The server's runtime is blocked while the POST is
    /// sent and the eviction fires, so both are pending when it wakes. Five
    /// rounds, each on a fresh connection.
    #[test]
    fn a_request_racing_an_eviction_is_answered() {
        let client = tokio::runtime::Runtime::new().unwrap();
        for round in 0..5 {
            let server = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let std_l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            std_l.set_nonblocking(true).unwrap();
            let addr = std_l.local_addr().unwrap();
            let ids = CallbackIds::default();
            let id = new_callback_id();
            ids.insert(&id, "ue3");
            let (tx, mut rx) = mpsc::channel(4);
            let st = ConnState {
                busy: Arc::new(AtomicUsize::new(0)),
                opening: Arc::new(AtomicBool::new(true)),
                last_active: Arc::new(Mutex::new(Instant::now())),
                evict: Arc::new(Notify::new()),
                draining: Arc::new(Mutex::new(None)),
            };
            let (busy, evict) = (st.busy.clone(), st.evict.clone());
            let handle = server.handle().clone();
            let served = std::thread::spawn(move || {
                server.block_on(async move {
                    let l = TcpListener::from_std(std_l).unwrap();
                    let (tcp, _) = l.accept().await.unwrap();
                    let _ = serve_connection(tcp, ids, tx, &st).await;
                });
            });
            let status = client.block_on(async {
                let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
                let (mut send, conn) = h2::client::handshake(tcp).await.unwrap();
                tokio::spawn(conn);
                // Warm-up: past the handshake and the opening window.
                let get = http::Request::get(format!("http://{addr}/other"))
                    .body(())
                    .unwrap();
                let (resp, _) = send.send_request(get, true).unwrap();
                assert_eq!(resp.await.unwrap().status(), StatusCode::NOT_FOUND);
                tokio::time::sleep(Duration::from_millis(20)).await;
                assert_eq!(busy.load(Ordering::SeqCst), 0);
                // Block the server, send the request, evict, then let it run.
                handle.spawn(async { std::thread::sleep(Duration::from_millis(300)) });
                tokio::time::sleep(Duration::from_millis(50)).await;
                let post = http::Request::post(format!("http://{addr}{CALLBACK_PREFIX}{id}"))
                    .body(())
                    .unwrap();
                let body = json!({
                    "subscription": "/3gpp-monitoring-event/v1/fl-af/subscriptions/s1",
                    "monitoringEventReports": [{
                        "monitoringType": "LOSS_OF_CONNECTIVITY",
                        "lossOfConnectReason": "DEREGISTERED",
                    }],
                });
                let (resp, mut stream) = send.send_request(post, false).unwrap();
                stream
                    .send_data(Bytes::from(body.to_string()), true)
                    .unwrap();
                tokio::time::sleep(Duration::from_millis(50)).await;
                evict.notify_one();
                resp.await.map(|r| r.status()).map_err(|e| e.to_string())
            });
            assert_eq!(status, Ok(StatusCode::NO_CONTENT), "round {round}");
            assert_eq!(rx.try_recv().map(|(i, _)| i).ok(), Some(id.clone()));
            served.join().unwrap();
        }
    }

    /// MINOR-2 (#531 round 4): the last window of a refusal burst is reported
    /// on a timer, not only when a later refusal comes.
    #[tokio::test]
    async fn the_last_refusal_window_is_flushed_on_a_timer() {
        let logs = Logs::default();
        let sub = tracing_subscriber::fmt()
            .with_writer(logs.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::WARN)
            .finish();
        let _guard = tracing::subscriber::set_default(sub);
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let (tx, _rx) = mpsc::channel(4);
        tokio::spawn(serve_callbacks_with(
            l,
            CallbackIds::default(),
            tx,
            Duration::from_millis(200),
        ));
        // Fill the source's slots with connections that stay opening (raw TCP,
        // no h2), then three more: refused. The first refusal warns at once;
        // the other two are only in the window.
        let mut held = Vec::new();
        for _ in 0..MAX_CONNECTIONS_PER_SOURCE {
            held.push(tcp_from(40, addr).await.unwrap());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        for _ in 0..3 {
            held.push(tcp_from(40, addr).await.unwrap());
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(600)).await;
        let text = logs.text();
        assert!(
            text.contains("1 connection(s) refused"),
            "the first refusal warns at once: {text}"
        );
        assert!(
            text.contains("2 connection(s) refused"),
            "the burst's last window was never reported: {text}"
        );
    }

    /// A tracing writer into a buffer.
    #[derive(Clone, Default)]
    struct Logs(Arc<Mutex<Vec<u8>>>);

    impl Logs {
        fn text(&self) -> String {
            String::from_utf8_lossy(&lock(&self.0)).into_owned()
        }
    }

    impl std::io::Write for Logs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            lock(&self.0).extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Logs {
        type Writer = Logs;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// A connection from `127.0.0.octet` to `addr`.
    async fn tcp_from(octet: u8, addr: SocketAddr) -> std::io::Result<tokio::net::TcpStream> {
        let s = tokio::net::TcpSocket::new_v4()?;
        s.bind(SocketAddr::from(([127, 0, 0, octet], 0)))?;
        s.connect(addr).await
    }

    /// One notification the way nefd sends it: a fresh h2c connection (from
    /// `127.0.0.octet`) per notification. The AF's status, or why it failed.
    async fn notify_from(octet: u8, addr: SocketAddr, id: &str) -> Result<StatusCode, String> {
        let tcp = tcp_from(octet, addr).await.map_err(|e| e.to_string())?;
        let (mut send, conn) = h2::client::handshake(tcp)
            .await
            .map_err(|e| e.to_string())?;
        let driver = tokio::spawn(conn);
        let body = json!({
            "subscription": "/3gpp-monitoring-event/v1/fl-af/subscriptions/s1",
            "monitoringEventReports": [{
                "monitoringType": "LOSS_OF_CONNECTIVITY",
                "lossOfConnectReason": "DEREGISTERED",
            }],
        });
        let req = http::Request::post(format!("http://{addr}{CALLBACK_PREFIX}{id}"))
            .header("content-type", "application/json")
            .body(())
            .unwrap();
        let (resp, mut stream) = send.send_request(req, false).map_err(|e| e.to_string())?;
        stream
            .send_data(Bytes::from(body.to_string()), true)
            .map_err(|e| e.to_string())?;
        let status = resp.await.map_err(|e| e.to_string())?.status();
        driver.abort();
        Ok(status)
    }

    async fn listener() -> (SocketAddr, String, mpsc::Receiver<(String, Bytes)>) {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let ids = CallbackIds::default();
        let id = new_callback_id();
        ids.insert(&id, "ue3");
        let (tx, rx) = mpsc::channel(1024);
        tokio::spawn(serve_callbacks(l, ids, tx));
        (addr, id, rx)
    }

    /// Idle peers cannot starve the NEF: one source opening far more idle h2c
    /// connections than its share keeps only [`MAX_CONNECTIONS_PER_SOURCE`], by
    /// evicting its own oldest, and a notification from another source is
    /// answered 204 and forwarded.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idle_connections_do_not_starve_a_notification() {
        let (addr, id, mut rx) = listener().await;
        let mut idle = Vec::new();
        for _ in 0..MAX_CALLBACK_CONNECTIONS + 8 {
            let tcp = tcp_from(20, addr).await.unwrap();
            let (mut send, conn) = h2::client::handshake(tcp).await.unwrap();
            let driver = tokio::spawn(conn);
            // One request each: past its opening, so it is idle and evictable.
            let req = http::Request::get(format!("http://{addr}/other"))
                .body(())
                .unwrap();
            let (resp, _) = send.send_request(req, true).unwrap();
            assert_eq!(resp.await.unwrap().status(), StatusCode::NOT_FOUND);
            idle.push((send, driver));
        }
        assert_eq!(notify_from(2, addr, &id).await, Ok(StatusCode::NO_CONTENT));
        let (got, _) = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got, id);
        let (_, first) = idle.remove(0);
        tokio::time::timeout(Duration::from_secs(5), first)
            .await
            .expect("the source's own oldest idle connection was evicted")
            .unwrap()
            .unwrap();
    }

    /// Smoke test, not the proof: 64 tasks on one source address connect raw
    /// TCP in a loop, and 200 notifications from another source, each on a
    /// fresh connection like nefd's, must all be answered 204. It is
    /// statistical and does not catch every revert of the eviction rules (the
    /// round-1 policy passes it); the deterministic Registry tests above carry
    /// that. Ignored in the default suite because the flood loads loopback for
    /// the other tests: run it alone, serially (`--ignored --test-threads=1`),
    /// as CI and `scripts/demo/d445a-reverts.sh` do.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "smoke test; run serially with --ignored --test-threads=1"]
    async fn a_churning_source_cannot_evict_another_sources_notification() {
        churn_against_notifications(1).await;
    }

    /// Smoke test, as above, with the flood spread over 20 source addresses
    /// (160 tasks, more than the listener holds in all) so it sits at its total
    /// cap. The deterministic Registry tests are the proof.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "smoke test; run serially with --ignored --test-threads=1"]
    async fn a_churning_flood_from_many_sources_cannot_evict_a_notification() {
        churn_against_notifications(20).await;
    }

    async fn churn_against_notifications(sources: u8) {
        let (addr, id, mut rx) = listener().await;
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Each task connects raw TCP, holds the connection until the listener
        // drops it (or for 100-300 ms), and connects again: the listener is
        // kept at its caps and evicting all the time. The flood runs on its own
        // runtime, as another host would, so it loads the listener and not the
        // test's own scheduler.
        let tasks = 64u16.max(8 * u16::from(sources));
        let flooding = stop.clone();
        let flood = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            for t in 0..tasks {
                let stop = flooding.clone();
                let octet = 20 + (t % u16::from(sources)) as u8;
                rt.spawn(async move {
                    while !stop.load(Ordering::Relaxed) {
                        if let Ok(mut s) = tcp_from(octet, addr).await {
                            let mut buf = [0u8; 64];
                            let hold = Duration::from_millis(100 + u64::from(t % 200));
                            let _ = tokio::time::timeout(hold, async {
                                while !matches!(s.read(&mut buf).await, Ok(0) | Err(_)) {}
                            })
                            .await;
                        }
                        // Paced (about 1000 connects/s in all), so the flood
                        // loads the listener's slots, not the kernel's SYN
                        // queue: a SYN flood is the kernel's to absorb.
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                });
            }
            rt.block_on(async {
                while !flooding.load(Ordering::Relaxed) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            });
            rt.shutdown_background();
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut failed = Vec::new();
        for n in 0..200 {
            match notify_from(2, addr, &id).await {
                Ok(StatusCode::NO_CONTENT) => {}
                other => failed.push(format!("{n}: {other:?}")),
            }
        }
        stop.store(true, Ordering::Relaxed);
        flood.join().unwrap();
        assert!(
            failed.is_empty(),
            "{} of 200 failed: {failed:?}",
            failed.len()
        );
        let mut forwarded = 0;
        while rx.try_recv().is_ok() {
            forwarded += 1;
        }
        assert_eq!(forwarded, 200);
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
