//! A small h2c SBI client for gNB and UE 6G tasks that call nextgcore NFs
//! (nwdafd, NEF, eesd, amfd, NRF), nextgsim#457.
//!
//! - **Transport:** HTTP/2 with prior knowledge over cleartext TCP (h2c) for
//!   `http://`, the way nextgcore's SBI listeners speak it; HTTP/2 over TLS
//!   (ALPN `h2`) for `https://`, as nextgcore's NRF serves NF management
//!   (nextgcore#1617). TLS trusts only the deployment's CA and presents a
//!   client certificate when one is configured, from the
//!   `NEXTGCORE_NRF_TLS_*` variables nextgcore's NFs read. See [`tls`]. An
//!   `https://` URI without [`SbiClientConfig::tls`] is refused before
//!   anything is dialled.
//! - **JSON** bodies and a **timeout** on every attempt.
//! - **Retries only on idempotent methods** (GET, HEAD, PUT, DELETE, OPTIONS;
//!   RFC 9110 §9.2.2), on a transport error, a timeout, or a 502/503/504. A POST
//!   is sent exactly once, so a subscription is never duplicated.
//! - **Optional OAuth2** client credentials from the NRF (TS 29.510
//!   Nnrf_AccessToken; TS 33.501 §13.4.1), matching nextgcore's
//!   `docker/rust/docker-compose.oauth2.yml` profile: the token request carries an
//!   ES256 Client Credentials Assertion (TS 33.501 §13.3.8) in the `cca` field.
//!   See [`oauth`].
//! - **Fail closed:** with OAuth2 configured, a call either carries a token or is
//!   not sent. A token the NRF refuses, or a call that names no scope, is an
//!   error, never a silent tokenless call. Without OAuth2 configured, a producer's
//!   401 is reported as [`SbiError::Unauthorized`] and never retried.
//! - **No secret is logged:** tokens and keys have redacting `Debug`, and the
//!   client logs only method, path and status.
//!
//! The crate is feature-gated at its consumers (`sbi-client` on nextgsim-gnb and
//! nextgsim-ue), so a default gNB or UE build pulls in no HTTP/2 stack.

pub mod nrf;
pub mod oauth;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod tls;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use h2::client::SendRequest;
use http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode, Uri};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::net::TcpStream;
use tokio::sync::Mutex;

pub use oauth::{AccessToken, CcaKey, OAuth2Config, TOKEN_PATH};
pub use tls::NrfTls;

/// Why an SBI call failed.
#[derive(Debug, thiserror::Error)]
pub enum SbiError {
    /// The URI does not parse or has no authority.
    #[error("invalid SBI URI {0:?}")]
    InvalidUri(String),
    /// Only `http://` (h2c) and `https://` are supported.
    #[error("unsupported URI scheme {0:?}: this client speaks http:// (h2c) and https:// only")]
    UnsupportedScheme(String),
    /// An `https://` URI, but the client has no TLS settings
    /// ([`SbiClientConfig::tls`]): refused, never sent in the clear or
    /// against public roots.
    #[error("https:// to {0} needs TLS settings (the NEXTGCORE_NRF_TLS_* variables); none are configured")]
    TlsNotConfigured(String),
    /// The TLS settings are unusable (half a client pair, an unreadable CA or
    /// key), or the TLS handshake failed.
    #[error("TLS: {0}")]
    Tls(String),
    /// The TCP connection could not be made.
    #[error("connect to {authority} failed: {source}")]
    Connect {
        /// `host:port`.
        authority: String,
        /// The socket error.
        source: std::io::Error,
    },
    /// An HTTP/2 protocol or stream error.
    #[error("HTTP/2 error: {0}")]
    Http2(#[from] h2::Error),
    /// The request could not be built.
    #[error("request build error: {0}")]
    Http(#[from] http::Error),
    /// No answer within the configured timeout (on the last attempt).
    #[error("no answer within {0:?}")]
    Timeout(Duration),
    /// The response body exceeded the configured limit.
    #[error("response body exceeds {0} bytes")]
    BodyTooLarge(usize),
    /// The producer answered 401. With OAuth2 configured the cached token is
    /// dropped; the call is never retried tokenless.
    #[error("401 Unauthorized from the producer (WWW-Authenticate: {www_authenticate:?})")]
    Unauthorized {
        /// The producer's `WWW-Authenticate` header, if any.
        www_authenticate: Option<String>,
    },
    /// The producer answered 403.
    #[error("403 Forbidden from the producer")]
    Forbidden,
    /// A JSON helper got a non-2xx answer other than 401/403.
    #[error("unexpected status {status}")]
    Status {
        /// The status code.
        status: u16,
    },
    /// OAuth2 is configured but the call named no scope: refused rather than
    /// sent without a token.
    #[error("OAuth2 is required but the call names no scope; refusing a tokenless call")]
    ScopeRequired,
    /// The NRF refused the token request (its `error` code, if it sent one).
    #[error("the NRF refused the access token request: status {status}, error {error:?}")]
    TokenRefused {
        /// The NRF's status (400 for every TS 29.510 token error).
        status: u16,
        /// The `error` member (`invalid_client`, `invalid_scope`, ...).
        error: Option<String>,
    },
    /// The NRF's token response is not a usable bearer token.
    #[error("unusable access token response: {0}")]
    TokenInvalid(String),
    /// A body is not the JSON expected.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Key material could not be read, created or published.
    #[error("key error: {0}")]
    Key(String),
}

/// Result alias.
pub type SbiResult<T> = Result<T, SbiError>;

/// What token a call needs: the producer's NF type (the token audience) and
/// the service (the scope), e.g. `NWDAF` / `nnwdaf-analyticsinfo`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scope {
    /// `targetNfType` of the token request.
    pub target_nf_type: String,
    /// `scope` of the token request: the producer's service name.
    pub scope: String,
}

impl Scope {
    /// A scope for `scope` at producers of type `target_nf_type`.
    pub fn new(target_nf_type: impl Into<String>, scope: impl Into<String>) -> Self {
        Self {
            target_nf_type: target_nf_type.into(),
            scope: scope.into(),
        }
    }
}

/// Client settings.
#[derive(Debug, Clone)]
pub struct SbiClientConfig {
    /// Per-attempt timeout, connect included.
    pub timeout: Duration,
    /// Extra attempts for an idempotent method. A POST or PATCH gets none.
    pub retries: u32,
    /// Delay before the first retry; doubled for each later one.
    pub retry_backoff: Duration,
    /// Largest response body accepted.
    pub max_body: usize,
    /// NRF OAuth2. `None`: calls go without a token.
    pub oauth2: Option<OAuth2Config>,
    /// Bind every connection of this client, the token requests included, to
    /// this interface (`SO_BINDTODEVICE`, Linux), e.g. a UE's PDU session TUN
    /// `uesimtun0` (nextgsim#471). `None`: connect by the routing table.
    /// Bound, the URIs must name IP literals: name resolution would not go
    /// through the device, so a name is refused.
    pub bind_device: Option<String>,
    /// TLS for `https://` URIs (nextgcore#1617). `None`: an `https://` call is
    /// refused ([`SbiError::TlsNotConfigured`]). See
    /// [`SbiClientConfig::with_nrf_tls_from_env`].
    pub tls: Option<NrfTls>,
}

impl SbiClientConfig {
    /// This configuration with the TLS settings of the environment for the
    /// NRF at `nrf_uri` ([`NrfTls::from_env`]), logged when `nrf_uri` is
    /// `https://`. Every NRF consumer builds its clients through this.
    pub fn with_nrf_tls_from_env(mut self, nrf_uri: &str) -> Self {
        let tls = NrfTls::from_env(nrf_uri);
        tls.log(nrf_uri);
        self.tls = Some(tls);
        self
    }
}

impl Default for SbiClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            retries: 2,
            retry_backoff: Duration::from_millis(100),
            max_body: 1 << 20,
            oauth2: None,
            bind_device: None,
            tls: None,
        }
    }
}

/// An SBI answer.
#[derive(Debug, Clone)]
pub struct SbiResponse {
    /// The status.
    pub status: StatusCode,
    /// The headers.
    pub headers: HeaderMap,
    /// The body.
    pub body: Bytes,
}

impl SbiResponse {
    /// The body as JSON.
    pub fn json<T: DeserializeOwned>(&self) -> SbiResult<T> {
        Ok(serde_json::from_slice(&self.body)?)
    }
}

/// RFC 9110 §9.2.2 idempotent methods: the only ones this client retries.
pub fn is_idempotent(method: &Method) -> bool {
    matches!(
        *method,
        Method::GET | Method::HEAD | Method::PUT | Method::DELETE | Method::OPTIONS
    )
}

fn retryable_status(s: StatusCode) -> bool {
    matches!(s.as_u16(), 502..=504)
}

/// One producer authority's pooled connection. Each authority has its own
/// lock, so a connect that hangs on one producer never delays calls to another.
type ConnSlot = Arc<Mutex<Option<SendRequest<Bytes>>>>;

/// The h2c SBI client. One HTTP/2 connection per producer authority is reused
/// across calls and replaced after a transport error.
pub struct SbiClient {
    config: SbiClientConfig,
    /// Held only to find or insert a slot, never across an await.
    conns: std::sync::Mutex<HashMap<String, ConnSlot>>,
    tokens: oauth::TokenCache,
    /// The rustls configuration of [`SbiClientConfig::tls`], built on the
    /// first `https://` call (or the error it gave, reported on every one).
    tls: std::sync::OnceLock<Result<Arc<rustls::ClientConfig>, String>>,
}

impl std::fmt::Debug for SbiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SbiClient")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

struct Target {
    /// `host:port` to dial.
    authority: String,
    /// The host as written (IPv6 in brackets).
    host: String,
    port: u16,
    https: bool,
    uri: Uri,
}

impl Target {
    /// `host:port` with the host lower-cased, to compare authorities.
    fn authority_key(&self) -> String {
        format!("{}:{}", self.host.to_ascii_lowercase(), self.port)
    }

    /// The connection-pool key: scheme and authority, so an `http://` and an
    /// `https://` call to one authority never share a connection.
    fn pool_key(&self) -> String {
        let scheme = if self.https { "https" } else { "http" };
        format!("{scheme}://{}", self.authority_key())
    }
}

fn parse_target(uri: &str) -> SbiResult<Target> {
    let parsed: Uri = uri
        .parse()
        .map_err(|_| SbiError::InvalidUri(uri.to_string()))?;
    let https = match parsed.scheme_str() {
        Some(s) if s.eq_ignore_ascii_case("http") => false,
        Some(s) if s.eq_ignore_ascii_case("https") => true,
        Some(other) => return Err(SbiError::UnsupportedScheme(other.to_string())),
        None => return Err(SbiError::InvalidUri(uri.to_string())),
    };
    let authority = parsed
        .authority()
        .ok_or_else(|| SbiError::InvalidUri(uri.to_string()))?;
    let host = authority.host().to_string();
    if host.is_empty() {
        return Err(SbiError::InvalidUri(uri.to_string()));
    }
    let port = authority.port_u16().unwrap_or(if https { 443 } else { 80 });
    Ok(Target {
        authority: format!("{host}:{port}"),
        host,
        port,
        https,
        uri: parsed,
    })
}

impl SbiClient {
    /// A client with `config`.
    pub fn new(config: SbiClientConfig) -> Self {
        Self {
            config,
            conns: std::sync::Mutex::new(HashMap::new()),
            tokens: oauth::TokenCache::default(),
            tls: std::sync::OnceLock::new(),
        }
    }

    /// The settings this client was built with.
    pub fn config(&self) -> &SbiClientConfig {
        &self.config
    }

    /// `GET` `uri` and decode a 2xx JSON body.
    pub async fn get_json<T: DeserializeOwned>(
        &self,
        uri: &str,
        scope: Option<&Scope>,
    ) -> SbiResult<T> {
        let r = self.request(Method::GET, uri, None, scope).await?;
        expect_2xx(&r)?;
        r.json()
    }

    /// `POST` a JSON body to `uri`, exactly once.
    pub async fn post_json<B: Serialize>(
        &self,
        uri: &str,
        body: &B,
        scope: Option<&Scope>,
    ) -> SbiResult<SbiResponse> {
        let body = Bytes::from(serde_json::to_vec(body)?);
        self.request(Method::POST, uri, Some(body), scope).await
    }

    /// `PUT` a JSON body to `uri` (retried, being idempotent).
    pub async fn put_json<B: Serialize>(
        &self,
        uri: &str,
        body: &B,
        scope: Option<&Scope>,
    ) -> SbiResult<SbiResponse> {
        let body = Bytes::from(serde_json::to_vec(body)?);
        self.request(Method::PUT, uri, Some(body), scope).await
    }

    /// `PATCH` an RFC 7396 JSON merge patch to `uri` (`application/merge-patch+json`),
    /// exactly once (PATCH is not idempotent).
    pub async fn patch_merge_json<B: Serialize>(
        &self,
        uri: &str,
        body: &B,
        scope: Option<&Scope>,
    ) -> SbiResult<SbiResponse> {
        let body = Bytes::from(serde_json::to_vec(body)?);
        self.request_typed(
            Method::PATCH,
            uri,
            Some(body),
            HeaderValue::from_static("application/merge-patch+json"),
            scope,
        )
        .await
    }

    /// `DELETE` `uri` (retried, being idempotent).
    pub async fn delete(&self, uri: &str, scope: Option<&Scope>) -> SbiResult<SbiResponse> {
        self.request(Method::DELETE, uri, None, scope).await
    }

    /// Send one request with a JSON body (or none).
    ///
    /// Returns the answer for any status except 401 ([`SbiError::Unauthorized`])
    /// and 403 ([`SbiError::Forbidden`]). With OAuth2 configured, `scope` is
    /// mandatory and a token is fetched (or reused) before anything is sent; if
    /// that fails, nothing is sent.
    pub async fn request(
        &self,
        method: Method,
        uri: &str,
        body: Option<Bytes>,
        scope: Option<&Scope>,
    ) -> SbiResult<SbiResponse> {
        self.request_typed(
            method,
            uri,
            body,
            HeaderValue::from_static("application/json"),
            scope,
        )
        .await
    }

    /// [`Self::request`] with the body's `content-type`.
    pub async fn request_typed(
        &self,
        method: Method,
        uri: &str,
        body: Option<Bytes>,
        body_type: HeaderValue,
        scope: Option<&Scope>,
    ) -> SbiResult<SbiResponse> {
        let target = parse_target(uri)?;
        let token = match (&self.config.oauth2, scope) {
            (Some(o), Some(s)) => Some(self.tokens.get(self, o, s).await?),
            (Some(_), None) => return Err(SbiError::ScopeRequired),
            (None, _) => None,
        };
        let content_type = body.as_ref().map(|_| body_type);
        let resp = self
            .send_with_retries(&method, &target, body, content_type, token.as_ref())
            .await?;
        match resp.status {
            StatusCode::UNAUTHORIZED => {
                if let (Some(_), Some(s)) = (&token, scope) {
                    self.tokens.invalidate(s).await;
                }
                Err(SbiError::Unauthorized {
                    www_authenticate: resp
                        .headers
                        .get(header::WWW_AUTHENTICATE)
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_string),
                })
            }
            StatusCode::FORBIDDEN => Err(SbiError::Forbidden),
            _ => Ok(resp),
        }
    }

    async fn send_with_retries(
        &self,
        method: &Method,
        target: &Target,
        body: Option<Bytes>,
        content_type: Option<HeaderValue>,
        token: Option<&AccessToken>,
    ) -> SbiResult<SbiResponse> {
        let attempts = if is_idempotent(method) {
            1 + self.config.retries
        } else {
            1
        };
        let mut backoff = self.config.retry_backoff;
        let mut attempt = 0;
        loop {
            attempt += 1;
            let last = attempt >= attempts;
            let outcome = tokio::time::timeout(
                self.config.timeout,
                self.send_once(method, target, body.clone(), content_type.clone(), token),
            )
            .await;
            let retry_reason = match outcome {
                Ok(Ok(resp)) if !last && retryable_status(resp.status) => {
                    format!("status {}", resp.status.as_u16())
                }
                Ok(Ok(resp)) => return Ok(resp),
                Ok(Err(e)) => {
                    // A transport failure may have left the connection unusable.
                    self.evict(&target.pool_key()).await;
                    if last || !transient(&e) {
                        return Err(e);
                    }
                    e.to_string()
                }
                Err(_) => {
                    self.evict(&target.pool_key()).await;
                    if last {
                        return Err(SbiError::Timeout(self.config.timeout));
                    }
                    "timeout".to_string()
                }
            };
            tracing::debug!(
                "SBI {method} {}: attempt {attempt}/{attempts} failed ({retry_reason}); retrying",
                target.uri.path()
            );
            tokio::time::sleep(backoff).await;
            backoff = backoff.saturating_mul(2);
        }
    }

    fn slot(&self, authority: &str) -> ConnSlot {
        self.conns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(authority.to_string())
            .or_default()
            .clone()
    }

    async fn evict(&self, authority: &str) {
        let slot = self.slot(authority);
        *slot.lock().await = None;
    }

    /// The TLS connector and server name for `target`, checked before
    /// anything is dialled.
    fn tls_for(
        &self,
        target: &Target,
    ) -> SbiResult<(
        tokio_rustls::TlsConnector,
        rustls::pki_types::ServerName<'static>,
    )> {
        let Some(settings) = &self.config.tls else {
            return Err(SbiError::TlsNotConfigured(target.authority.clone()));
        };
        let config = self
            .tls
            .get_or_init(|| settings.client_config())
            .clone()
            .map_err(SbiError::Tls)?;
        let name =
            tls::server_name(settings.server_name_for(&target.authority_key(), &target.host))?;
        Ok((tokio_rustls::TlsConnector::from(config), name))
    }

    /// The pooled connection to `target`, or a new one; `true` when pooled.
    async fn connection(&self, target: &Target) -> SbiResult<(SendRequest<Bytes>, bool)> {
        let slot = self.slot(&target.pool_key());
        let mut pooled = slot.lock().await;
        if let Some(c) = pooled.as_ref() {
            return Ok((c.clone(), true));
        }
        let tls = target.https.then(|| self.tls_for(target)).transpose()?;
        let authority = &target.authority;
        let tcp = connect(authority, self.config.bind_device.as_deref())
            .await
            .map_err(|source| SbiError::Connect {
                authority: authority.to_string(),
                source,
            })?;
        let _ = tcp.set_nodelay(true);
        let send = match tls {
            None => {
                let (send, conn) = h2::client::handshake(tcp).await?;
                tokio::spawn(async move {
                    if let Err(e) = conn.await {
                        tracing::debug!("SBI h2c connection closed: {e}");
                    }
                });
                send
            }
            Some((connector, name)) => {
                let stream = connector
                    .connect(name, tcp)
                    .await
                    .map_err(|e| SbiError::Tls(format!("handshake with {authority}: {e}")))?;
                if let Some(alpn) = stream.get_ref().1.alpn_protocol() {
                    if alpn != b"h2" {
                        return Err(SbiError::Tls(format!(
                            "{authority} negotiated ALPN {:?}, not h2",
                            String::from_utf8_lossy(alpn)
                        )));
                    }
                }
                let (send, conn) = h2::client::handshake(stream).await?;
                tokio::spawn(async move {
                    if let Err(e) = conn.await {
                        tracing::debug!("SBI h2 (TLS) connection closed: {e}");
                    }
                });
                send
            }
        };
        *pooled = Some(send.clone());
        Ok((send, false))
    }

    async fn send_once(
        &self,
        method: &Method,
        target: &Target,
        body: Option<Bytes>,
        content_type: Option<HeaderValue>,
        token: Option<&AccessToken>,
    ) -> SbiResult<SbiResponse> {
        let (send, pooled) = self.connection(target).await?;
        let mut send = match send.ready().await {
            Ok(s) => s,
            // A pooled connection the peer has since closed: nothing was sent
            // on it, so one fresh connection is safe even for a POST.
            Err(e) if pooled => {
                tracing::debug!(
                    "SBI pooled connection to {} is gone ({e}); reconnecting",
                    target.authority
                );
                self.evict(&target.pool_key()).await;
                let (fresh, _) = self.connection(target).await?;
                fresh.ready().await?
            }
            Err(e) => return Err(e.into()),
        };
        let mut req = Request::builder()
            .method(method.clone())
            .uri(target.uri.clone())
            .header(header::ACCEPT, "application/json, application/problem+json");
        if let Some(ct) = content_type {
            req = req.header(header::CONTENT_TYPE, ct);
        }
        if let Some(t) = token {
            req = req.header(header::AUTHORIZATION, t.header_value()?);
        }
        let req = req.body(())?;
        let end_of_stream = body.as_ref().is_none_or(Bytes::is_empty);
        let (resp, mut stream) = send.send_request(req, end_of_stream)?;
        if let Some(mut data) = body.filter(|b| !b.is_empty()) {
            // RFC 9113 §8.1: a server may send a complete response before the
            // request is complete, then RST_STREAM(NO_ERROR) the rest of it; the
            // client "MUST NOT discard responses as a result of receiving such
            // a RST_STREAM". h2 then reports the stream as unable to send
            // (`poll_capacity` is `None`, or `send_data` fails). Stop sending and
            // read the response: it is the server's answer, or, if the stream
            // was reset with no response, the reset is the error (#552; this was
            // a synthetic CANCEL that dropped the answer).
            while !data.is_empty() {
                stream.reserve_capacity(data.len());
                let Some(Ok(cap)) = std::future::poll_fn(|cx| stream.poll_capacity(cx)).await
                else {
                    break;
                };
                let chunk = data.split_to(cap.min(data.len()));
                if stream.send_data(chunk, data.is_empty()).is_err() {
                    break;
                }
            }
        }
        let resp = resp.await?;
        let (parts, mut recv) = resp.into_parts();
        let mut buf = BytesMut::new();
        while let Some(chunk) = recv.data().await {
            let chunk = chunk?;
            let _ = recv.flow_control().release_capacity(chunk.len());
            if buf.len() + chunk.len() > self.config.max_body {
                return Err(SbiError::BodyTooLarge(self.config.max_body));
            }
            buf.extend_from_slice(&chunk);
        }
        tracing::debug!(
            "SBI {method} {} -> {}",
            target.uri.path(),
            parts.status.as_u16()
        );
        Ok(SbiResponse {
            status: parts.status,
            headers: parts.headers,
            body: buf.freeze(),
        })
    }
}

/// A TCP connection to `authority` (`host:port`), bound to `device` if given.
async fn connect(authority: &str, device: Option<&str>) -> std::io::Result<TcpStream> {
    let Some(device) = device else {
        return TcpStream::connect(authority).await;
    };
    // A name would be resolved by the host's resolver, outside the device, so
    // a bound client takes IP literals only (nextgsim#471 review m3).
    let addr: std::net::SocketAddr = authority.parse().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "bound to {device}: {authority} must be an IP literal (a name would be resolved \
                 outside the device)"
            ),
        )
    })?;
    let socket = if addr.is_ipv4() {
        tokio::net::TcpSocket::new_v4()?
    } else {
        tokio::net::TcpSocket::new_v6()?
    };
    bind_to_device(&socket, device)?;
    socket.connect(addr).await
}

#[cfg(any(target_os = "android", target_os = "fuchsia", target_os = "linux"))]
fn bind_to_device(socket: &tokio::net::TcpSocket, device: &str) -> std::io::Result<()> {
    socket.bind_device(Some(device.as_bytes()))
}

#[cfg(not(any(target_os = "android", target_os = "fuchsia", target_os = "linux")))]
fn bind_to_device(_socket: &tokio::net::TcpSocket, device: &str) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("binding to {device} needs SO_BINDTODEVICE (Linux)"),
    ))
}

fn transient(e: &SbiError) -> bool {
    matches!(e, SbiError::Connect { .. } | SbiError::Http2(_))
}

fn expect_2xx(r: &SbiResponse) -> SbiResult<()> {
    if r.status.is_success() {
        Ok(())
    } else {
        Err(SbiError::Status {
            status: r.status.as_u16(),
        })
    }
}

#[cfg(test)]
mod tests;
