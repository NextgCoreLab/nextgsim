//! NRF OAuth2 client credentials (TS 29.510 §6.3 Nnrf_AccessToken, TS 33.501
//! §13.4.1), shaped like nextgcore's own NF client so it works against the
//! `docker/rust/docker-compose.oauth2.yml` profile unchanged:
//!
//! - `POST {nrf}/nnrf-oauth2/v1/access-token` ([`TOKEN_PATH`]), form-encoded:
//!   `grant_type=client_credentials`, `nfInstanceId`, `nfType`, `targetNfType`,
//!   `scope`, `cca`.
//! - `cca` is a Client Credentials Assertion (TS 33.501 §13.3.8): a JWT with
//!   header `{"alg":"ES256","typ":"JWT"}` and claims `sub` = `iss` = the
//!   requester's nfInstanceId, `aud` = `"NRF"`, `iat`, `exp` = `iat` + 60 s.
//! - The NRF trusts the requester's public key as `<nfInstanceId>.jwk` in its
//!   `cca_trusted_keys_dir`; [`CcaKey::publish_jwk`] writes that file. The
//!   requester must also be registered at the NRF under the same nfInstanceId
//!   and nfType.
//! - The key file is the hex-encoded raw P-256 scalar, created `0600` when
//!   absent, the format nextgcore's `load_or_create_es256_key` uses. A malformed
//!   file is an error, never a silent regeneration.
//!
//! Tokens are cached per (targetNfType, scope) and refreshed 30 s before
//! `expires_in`. The token request is a POST and is never retried.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use bytes::Bytes;
use http::{header, HeaderValue, Method, StatusCode};
use p256::ecdsa::signature::Signer as _;
use p256::ecdsa::{Signature, SigningKey};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::{parse_target, SbiClient, SbiError, SbiResult, Scope};

/// The token path nextgcore's NRF serves (its client default; the standard
/// `/oauth2/token` is not routed by nextgcore's NRF).
pub const TOKEN_PATH: &str = "/nnrf-oauth2/v1/access-token";

/// CCA lifetime, as nextgcore mints it.
pub const CCA_LIFETIME_SECS: u64 = 60;

/// Refresh a cached token this long before it expires.
const REFRESH_MARGIN: Duration = Duration::from_secs(30);

/// `expires_in` when the NRF omits it (RFC 6749 §5.1 makes it optional).
const DEFAULT_EXPIRES_IN: u64 = 3600;

/// An ES256 (P-256) CCA signing key. `Debug` never shows the key.
#[derive(Clone)]
pub struct CcaKey(pub(crate) SigningKey);

impl std::fmt::Debug for CcaKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CcaKey(<redacted>)")
    }
}

impl CcaKey {
    /// A fresh random key.
    pub fn generate() -> Self {
        Self(SigningKey::random(&mut rand::rngs::OsRng))
    }

    /// Load the hex-encoded scalar at `path`, or create one there (`0600`).
    /// A file that exists but does not hold a valid key is an error.
    pub fn load_or_create(path: &Path) -> SbiResult<Self> {
        if path.exists() {
            let text = std::fs::read_to_string(path)
                .map_err(|e| SbiError::Key(format!("read {}: {e}", path.display())))?;
            let raw = hex::decode(text.trim()).map_err(|e| {
                SbiError::Key(format!(
                    "{} is not hex ({e}); refusing to regenerate it",
                    path.display()
                ))
            })?;
            let key = SigningKey::from_slice(&raw).map_err(|e| {
                SbiError::Key(format!(
                    "{} does not hold a valid P-256 scalar: {e}",
                    path.display()
                ))
            })?;
            return Ok(Self(key));
        }
        let key = Self::generate();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| SbiError::Key(format!("create {}: {e}", parent.display())))?;
        }
        write_private(path, hex::encode(key.0.to_bytes()).as_bytes())?;
        Ok(key)
    }

    /// The public half as an RFC 7517 EC JWK with `kid` = `nf_instance_id`,
    /// the form nextgcore's NRF reads from its trusted-keys directory.
    pub fn public_jwk(&self, nf_instance_id: &str) -> serde_json::Value {
        let point = self.0.verifying_key().to_encoded_point(false);
        let (x, y) = point.as_bytes()[1..].split_at(32);
        serde_json::json!({
            "kty": "EC",
            "crv": "P-256",
            "alg": "ES256",
            "use": "sig",
            "kid": nf_instance_id,
            "x": URL_SAFE_NO_PAD.encode(x),
            "y": URL_SAFE_NO_PAD.encode(y),
        })
    }

    /// Write [`public_jwk`](Self::public_jwk) to `<dir>/<nf_instance_id>.jwk`
    /// (public, `0644`). Returns the path.
    pub fn publish_jwk(&self, dir: &Path, nf_instance_id: &str) -> SbiResult<std::path::PathBuf> {
        if nf_instance_id.is_empty()
            || nf_instance_id.contains(['/', '\\'])
            || nf_instance_id.contains("..")
        {
            return Err(SbiError::Key(format!(
                "nfInstanceId {nf_instance_id:?} is not a safe file name"
            )));
        }
        std::fs::create_dir_all(dir)
            .map_err(|e| SbiError::Key(format!("create {}: {e}", dir.display())))?;
        let path = dir.join(format!("{nf_instance_id}.jwk"));
        let text = serde_json::to_vec(&self.public_jwk(nf_instance_id))?;
        std::fs::write(&path, text)
            .map_err(|e| SbiError::Key(format!("write {}: {e}", path.display())))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644));
        }
        Ok(path)
    }

    /// The verifying key, for tests and stubs.
    pub fn verifying_key(&self) -> p256::ecdsa::VerifyingKey {
        *self.0.verifying_key()
    }

    /// A CCA for `nf_instance_id`, issued at `iat` (seconds since the epoch).
    pub fn mint_cca(&self, nf_instance_id: &str, iat: u64) -> String {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"ES256","typ":"JWT"}"#);
        let claims = serde_json::json!({
            "sub": nf_instance_id,
            "iss": nf_instance_id,
            "aud": "NRF",
            "iat": iat,
            "exp": iat + CCA_LIFETIME_SECS,
        });
        let payload = URL_SAFE_NO_PAD.encode(claims.to_string());
        let signing_input = format!("{header}.{payload}");
        let sig: Signature = self.0.sign(signing_input.as_bytes());
        format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(sig.to_bytes()))
    }
}

fn write_private(path: &Path, data: &[u8]) -> SbiResult<()> {
    #[cfg(unix)]
    {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| SbiError::Key(format!("create {}: {e}", path.display())))?;
        f.write_all(data)
            .map_err(|e| SbiError::Key(format!("write {}: {e}", path.display())))
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, data)
            .map_err(|e| SbiError::Key(format!("write {}: {e}", path.display())))
    }
}

/// NRF OAuth2 settings. With these set on a client, every call must name a
/// [`Scope`](crate::Scope) and carries a token, or is not sent.
#[derive(Debug, Clone)]
pub struct OAuth2Config {
    /// The NRF's API root, e.g. `http://172.23.0.10:7777`.
    pub nrf_api_root: String,
    /// This requester's nfInstanceId: the CCA's `sub`/`iss`, the JWK's `kid`,
    /// and the id it is registered under at the NRF.
    pub nf_instance_id: String,
    /// The nfType it is registered under (e.g. `AF`).
    pub nf_type: String,
    /// Its CCA signing key.
    pub cca_key: CcaKey,
    /// The token path; [`TOKEN_PATH`] for nextgcore.
    pub token_path: String,
}

impl OAuth2Config {
    /// Settings for nextgcore's NRF at `nrf_api_root`.
    pub fn new(
        nrf_api_root: impl Into<String>,
        nf_instance_id: impl Into<String>,
        nf_type: impl Into<String>,
        cca_key: CcaKey,
    ) -> Self {
        Self {
            nrf_api_root: nrf_api_root.into(),
            nf_instance_id: nf_instance_id.into(),
            nf_type: nf_type.into(),
            cca_key,
            token_path: TOKEN_PATH.to_string(),
        }
    }
}

/// A bearer token. `Debug` and `Display` never show it.
#[derive(Clone, PartialEq, Eq)]
pub struct AccessToken(pub(crate) String);

impl std::fmt::Debug for AccessToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AccessToken(<redacted>)")
    }
}

impl AccessToken {
    /// The `Authorization` header value, marked sensitive.
    pub(crate) fn header_value(&self) -> SbiResult<HeaderValue> {
        let mut v = HeaderValue::from_str(&format!("Bearer {}", self.0))
            .map_err(|_| SbiError::TokenInvalid("the token is not a valid header value".into()))?;
        v.set_sensitive(true);
        Ok(v)
    }

    /// The raw token, for a caller that must forward it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    token_type: String,
    expires_in: Option<u64>,
}

#[derive(Deserialize)]
struct TokenErrorBody {
    error: Option<String>,
}

struct Cached {
    token: AccessToken,
    refresh_at: Instant,
}

/// Tokens by scope. The lock is held across a fetch, so concurrent calls for
/// one scope cause one token request, not several.
#[derive(Default)]
pub(crate) struct TokenCache {
    entries: Mutex<HashMap<Scope, Cached>>,
}

/// `application/x-www-form-urlencoded` (WHATWG URL §5.2): space as `+`.
pub(crate) fn form_encode(pairs: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push('&');
        }
        enc(&mut out, k);
        out.push('=');
        enc(&mut out, v);
    }
    out
}

fn enc(out: &mut String, s: &str) {
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl TokenCache {
    pub(crate) async fn get(
        &self,
        client: &SbiClient,
        cfg: &OAuth2Config,
        scope: &Scope,
    ) -> SbiResult<AccessToken> {
        let mut entries = self.entries.lock().await;
        if let Some(c) = entries.get(scope) {
            if Instant::now() < c.refresh_at {
                return Ok(c.token.clone());
            }
        }
        let (token, expires_in) = fetch(client, cfg, scope).await?;
        let lifetime = Duration::from_secs(expires_in);
        entries.insert(
            scope.clone(),
            Cached {
                token: token.clone(),
                refresh_at: Instant::now() + lifetime.saturating_sub(REFRESH_MARGIN),
            },
        );
        tracing::debug!(
            "NRF granted a token for {} / {} (expires in {expires_in} s)",
            scope.target_nf_type,
            scope.scope
        );
        Ok(token)
    }

    pub(crate) async fn invalidate(&self, scope: &Scope) {
        self.entries.lock().await.remove(scope);
    }
}

async fn fetch(
    client: &SbiClient,
    cfg: &OAuth2Config,
    scope: &Scope,
) -> SbiResult<(AccessToken, u64)> {
    let uri = format!(
        "{}{}",
        cfg.nrf_api_root.trim_end_matches('/'),
        cfg.token_path
    );
    let target = parse_target(&uri)?;
    let cca = cfg.cca_key.mint_cca(&cfg.nf_instance_id, now_secs());
    let body = form_encode(&[
        ("grant_type", "client_credentials"),
        ("nfInstanceId", &cfg.nf_instance_id),
        ("nfType", &cfg.nf_type),
        ("targetNfType", &scope.target_nf_type),
        ("scope", &scope.scope),
        ("cca", &cca),
    ]);
    // A POST: send_with_retries sends it exactly once.
    let resp = client
        .send_with_retries(
            &Method::POST,
            &target,
            Some(Bytes::from(body)),
            Some(HeaderValue::from_static(
                "application/x-www-form-urlencoded",
            )),
            None,
        )
        .await?;
    if resp.status != StatusCode::OK {
        let error = serde_json::from_slice::<TokenErrorBody>(&resp.body)
            .ok()
            .and_then(|b| b.error);
        return Err(SbiError::TokenRefused {
            status: resp.status.as_u16(),
            error,
        });
    }
    let _ = resp.headers.get(header::CACHE_CONTROL);
    let t: TokenResponse = serde_json::from_slice(&resp.body)
        .map_err(|e| SbiError::TokenInvalid(format!("not a token response: {e}")))?;
    if !t.token_type.eq_ignore_ascii_case("bearer") {
        return Err(SbiError::TokenInvalid(format!(
            "token_type {:?} is not Bearer",
            t.token_type
        )));
    }
    if t.access_token.is_empty() {
        return Err(SbiError::TokenInvalid("empty access_token".into()));
    }
    Ok((
        AccessToken(t.access_token),
        t.expires_in.unwrap_or(DEFAULT_EXPIRES_IN),
    ))
}
