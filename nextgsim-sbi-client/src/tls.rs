//! `https://` toward the NRF (nextgcore#1617): TLS 1.2+ with the deployment's
//! CA as the only trust anchor and an optional client certificate.
//!
//! nextgcore's nrfd refuses NF management (NFRegister, NFUpdate,
//! NFDeregister) from a caller without a verified certificate whose URI SAN
//! names the nfInstanceId it registers (TS 33.310 §6.1.3c.3; TS 33.501
//! §13.1.0: the NRF authenticates its consumers over TLS). The settings come
//! from the same environment variables nextgcore's NFs read
//! (`nextgcore_sbi::nrf_tls`), so a deployment sets one set of names for the
//! core and for nextgsim:
//!
//! | Setting | Variable | Fallback |
//! |---|---|---|
//! | CA (trust anchor) | `NEXTGCORE_NRF_TLS_CA` | `NEXTGCORE_SBI_TLS_CA`, then `/etc/nextgcore/tls/ca.crt` |
//! | client certificate | `NEXTGCORE_NRF_TLS_CLIENT_CERT` | `NEXTGCORE_SBI_TLS_CLIENT_CERT` |
//! | client key | `NEXTGCORE_NRF_TLS_CLIENT_KEY` | `NEXTGCORE_SBI_TLS_CLIENT_KEY` |
//! | server name | `NEXTGCORE_NRF_TLS_SERVER_NAME` | the URI's host |
//!
//! - **No public roots.** The CA file is the only trust anchor; a server
//!   certificate it did not issue fails the handshake.
//! - **Half a client pair is an error,** reported before anything is dialled:
//!   a certificate without its key (or the reverse) never silently becomes a
//!   call without a certificate.
//! - **The server name applies only to the NRF's own authority** (host and
//!   port of the NRF URI), so a client dialling the NRF by `127.0.0.1` can
//!   verify it as `DNS:nrf`, while any other `https` authority is verified as
//!   itself.
//! - Plain `http://` (h2c) is unchanged.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustls::pki_types::pem::PemObject as _;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};

use crate::{parse_target, SbiError, SbiResult};

/// The NRF's CA (nextgcore `NRF_TLS_CA_ENV`).
pub const NRF_TLS_CA_ENV: &str = "NEXTGCORE_NRF_TLS_CA";
/// The client certificate presented to the NRF.
pub const NRF_TLS_CLIENT_CERT_ENV: &str = "NEXTGCORE_NRF_TLS_CLIENT_CERT";
/// Its private key.
pub const NRF_TLS_CLIENT_KEY_ENV: &str = "NEXTGCORE_NRF_TLS_CLIENT_KEY";
/// The name the NRF's certificate is verified against, for the NRF authority.
pub const NRF_TLS_SERVER_NAME_ENV: &str = "NEXTGCORE_NRF_TLS_SERVER_NAME";
/// The SBI-wide CA, used when no NRF CA is set (nextgcore `TlsPaths::CA_ENV`).
pub const SBI_TLS_CA_ENV: &str = "NEXTGCORE_SBI_TLS_CA";
/// The SBI-wide client certificate (nextgcore `TlsPaths::CLIENT_CERT_ENV`).
pub const SBI_TLS_CLIENT_CERT_ENV: &str = "NEXTGCORE_SBI_TLS_CLIENT_CERT";
/// The SBI-wide client key (nextgcore `TlsPaths::CLIENT_KEY_ENV`).
pub const SBI_TLS_CLIENT_KEY_ENV: &str = "NEXTGCORE_SBI_TLS_CLIENT_KEY";
/// The CA when no variable names one (nextgcore `TlsPaths::default`).
pub const DEFAULT_CA: &str = "/etc/nextgcore/tls/ca.crt";

/// Every variable [`NrfTls::from_env`] reads.
pub const ENV_VARS: [&str; 7] = [
    NRF_TLS_CA_ENV,
    NRF_TLS_CLIENT_CERT_ENV,
    NRF_TLS_CLIENT_KEY_ENV,
    NRF_TLS_SERVER_NAME_ENV,
    SBI_TLS_CA_ENV,
    SBI_TLS_CLIENT_CERT_ENV,
    SBI_TLS_CLIENT_KEY_ENV,
];

/// Serialises the tests that set the `NEXTGCORE_NRF_TLS_*` /
/// `NEXTGCORE_SBI_TLS_*` variables, the process-global state
/// [`NrfTls::from_env`] reads. Take it through
/// [`crate::test_support::NrfTlsEnv`].
#[cfg(any(test, feature = "test-support"))]
pub static NRF_TLS_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// TLS settings for `https://` calls; see the module doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NrfTls {
    /// The only trust anchor.
    pub ca_cert: PathBuf,
    /// The client certificate chain (PEM), if any.
    pub client_cert: Option<PathBuf>,
    /// Its private key (PEM), if any.
    pub client_key: Option<PathBuf>,
    /// The name to verify the NRF's certificate against, instead of its host.
    pub server_name: Option<String>,
    /// The NRF authority (`host:port`, host lower-cased) the server name
    /// applies to. `None`: it applies to no authority.
    pub nrf_authority: Option<String>,
    /// A configuration error (half a client pair): every `https` call fails
    /// with it before dialling.
    pub error: Option<String>,
}

impl NrfTls {
    /// The settings from the environment, for the NRF at `nrf_uri`.
    pub fn from_env(nrf_uri: &str) -> Self {
        Self::resolve(nrf_uri, |var| {
            std::env::var(var)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
    }

    /// The settings from `env` (a variable lookup), for the NRF at `nrf_uri`.
    pub fn resolve(nrf_uri: &str, env: impl Fn(&str) -> Option<String>) -> Self {
        let ca_cert = env(NRF_TLS_CA_ENV)
            .or_else(|| env(SBI_TLS_CA_ENV))
            .unwrap_or_else(|| DEFAULT_CA.to_string());
        let pair = |cert_var: &str, key_var: &str| {
            let (c, k) = (env(cert_var), env(key_var));
            (c.is_some() || k.is_some()).then_some((c, k))
        };
        // The NRF pair wins as a whole; a half NRF pair is not completed from
        // the SBI one (nextgcore's precedence).
        let (client_cert, client_key) = pair(NRF_TLS_CLIENT_CERT_ENV, NRF_TLS_CLIENT_KEY_ENV)
            .or_else(|| pair(SBI_TLS_CLIENT_CERT_ENV, SBI_TLS_CLIENT_KEY_ENV))
            .unwrap_or((None, None));
        let error = match (&client_cert, &client_key) {
            (Some(c), None) => Some(format!(
                "NRF client TLS: certificate {c} is configured without its private key (set \
                 {NRF_TLS_CLIENT_KEY_ENV} or {SBI_TLS_CLIENT_KEY_ENV})"
            )),
            (None, Some(k)) => Some(format!(
                "NRF client TLS: private key {k} is configured without its certificate (set \
                 {NRF_TLS_CLIENT_CERT_ENV} or {SBI_TLS_CLIENT_CERT_ENV})"
            )),
            _ => None,
        };
        Self {
            ca_cert: ca_cert.into(),
            client_cert: client_cert.map(PathBuf::from),
            client_key: client_key.map(PathBuf::from),
            server_name: env(NRF_TLS_SERVER_NAME_ENV),
            nrf_authority: parse_target(nrf_uri).ok().map(|t| t.authority_key()),
            error,
        }
    }

    /// Settings with only a CA, for no particular NRF.
    pub fn with_ca(ca_cert: impl Into<PathBuf>) -> Self {
        Self {
            ca_cert: ca_cert.into(),
            client_cert: None,
            client_key: None,
            server_name: None,
            nrf_authority: None,
            error: None,
        }
    }

    /// These settings presenting the certificate `cert` with its key `key`.
    pub fn client_pair(mut self, cert: impl Into<PathBuf>, key: impl Into<PathBuf>) -> Self {
        self.client_cert = Some(cert.into());
        self.client_key = Some(key.into());
        self
    }

    /// These settings verifying the NRF at `nrf_uri` as `server_name`.
    pub fn nrf_server_name(mut self, nrf_uri: &str, server_name: impl Into<String>) -> Self {
        self.nrf_authority = parse_target(nrf_uri).ok().map(|t| t.authority_key());
        self.server_name = Some(server_name.into());
        self
    }

    /// The name to verify `host` (of `authority_key`, `host:port`) against.
    pub(crate) fn server_name_for<'a>(&'a self, authority_key: &str, host: &'a str) -> &'a str {
        match (&self.server_name, &self.nrf_authority) {
            (Some(name), Some(nrf)) if nrf == authority_key => name,
            _ => host,
        }
    }

    /// Log what an `https` NRF call will use (no key material): ERROR for a
    /// configuration error, WARN without a client certificate (the NRF then
    /// sees no verified identity and refuses NF management), INFO otherwise.
    pub fn log(&self, nrf_uri: &str) {
        if !is_https(nrf_uri) {
            return;
        }
        if let Some(e) = &self.error {
            tracing::error!("{e}: every request to the NRF at {nrf_uri} will fail");
        } else if let Some(cert) = &self.client_cert {
            tracing::info!(
                "NRF client TLS for {nrf_uri}: CA={} client certificate={} server name={}",
                self.ca_cert.display(),
                cert.display(),
                self.server_name.as_deref().unwrap_or("(the URI host)")
            );
        } else {
            tracing::warn!(
                "NRF client TLS for {nrf_uri}: CA={}, but NO client certificate: the NRF sees no \
                 verified identity for this client and refuses its NF management. Set \
                 {NRF_TLS_CLIENT_CERT_ENV} / {NRF_TLS_CLIENT_KEY_ENV}.",
                self.ca_cert.display()
            );
        }
    }

    /// The rustls client configuration: the CA as the only root, ALPN `h2`,
    /// the client pair if set.
    pub(crate) fn client_config(&self) -> Result<Arc<rustls::ClientConfig>, String> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        let mut roots = rustls::RootCertStore::empty();
        for cert in read_certs(&self.ca_cert)? {
            roots
                .add(cert)
                .map_err(|e| format!("CA {}: {e}", self.ca_cert.display()))?;
        }
        let builder = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("TLS versions: {e}"))?
        .with_root_certificates(roots);
        let mut config = match (&self.client_cert, &self.client_key) {
            (Some(c), Some(k)) => {
                let key = PrivateKeyDer::from_pem_file(k)
                    .map_err(|e| format!("client key {}: {e}", k.display()))?;
                builder
                    .with_client_auth_cert(read_certs(c)?, key)
                    .map_err(|e| format!("client certificate {}: {e}", c.display()))?
            }
            _ => builder.with_no_client_auth(),
        };
        config.alpn_protocols = vec![b"h2".to_vec()];
        Ok(Arc::new(config))
    }
}

fn read_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>, String> {
    let certs = CertificateDer::pem_file_iter(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if certs.is_empty() {
        return Err(format!("{}: no PEM certificate", path.display()));
    }
    Ok(certs)
}

/// `uri` names the `https` scheme.
pub fn is_https(uri: &str) -> bool {
    uri.trim()
        .get(..8)
        .is_some_and(|s| s.eq_ignore_ascii_case("https://"))
}

/// The TLS server name for `name` (a DNS name or an IP literal, brackets
/// allowed).
pub(crate) fn server_name(name: &str) -> SbiResult<ServerName<'static>> {
    let bare = name.trim_start_matches('[').trim_end_matches(']');
    ServerName::try_from(bare.to_string())
        .map_err(|_| SbiError::Tls(format!("{name:?} is not a valid TLS server name")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn resolve(nrf: &str, env: &[(&str, &str)]) -> NrfTls {
        let env: HashMap<String, String> = env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        NrfTls::resolve(nrf, |v| env.get(v).cloned())
    }

    #[test]
    fn nrf_settings_win_over_the_sbi_ones_1617() {
        let t = resolve(
            "https://NRF:7777",
            &[
                (NRF_TLS_CA_ENV, "/n/ca.crt"),
                (NRF_TLS_CLIENT_CERT_ENV, "/n/gnb.crt"),
                (NRF_TLS_CLIENT_KEY_ENV, "/n/gnb.key"),
                (NRF_TLS_SERVER_NAME_ENV, "nrf"),
                (SBI_TLS_CA_ENV, "/s/ca.crt"),
                (SBI_TLS_CLIENT_CERT_ENV, "/s/c.crt"),
                (SBI_TLS_CLIENT_KEY_ENV, "/s/c.key"),
            ],
        );
        assert_eq!(t.ca_cert, PathBuf::from("/n/ca.crt"));
        assert_eq!(t.client_cert, Some("/n/gnb.crt".into()));
        assert_eq!(t.client_key, Some("/n/gnb.key".into()));
        assert_eq!(t.server_name.as_deref(), Some("nrf"));
        assert_eq!(t.nrf_authority.as_deref(), Some("nrf:7777"));
        assert_eq!(t.error, None);
    }

    #[test]
    fn sbi_settings_then_the_default_ca_1617() {
        let t = resolve(
            "https://nrf",
            &[
                (SBI_TLS_CA_ENV, "/s/ca.crt"),
                (SBI_TLS_CLIENT_CERT_ENV, "/s/c.crt"),
                (SBI_TLS_CLIENT_KEY_ENV, "/s/c.key"),
            ],
        );
        assert_eq!(t.ca_cert, PathBuf::from("/s/ca.crt"));
        assert_eq!(t.client_cert, Some("/s/c.crt".into()));
        assert_eq!(t.nrf_authority.as_deref(), Some("nrf:443"));
        let t = resolve("https://nrf", &[]);
        assert_eq!(t.ca_cert, PathBuf::from(DEFAULT_CA));
        assert_eq!((t.client_cert, t.client_key, t.error), (None, None, None));
    }

    #[test]
    fn half_a_pair_is_an_error_never_completed_from_the_other_pair_1617() {
        let t = resolve(
            "https://nrf",
            &[
                (NRF_TLS_CLIENT_CERT_ENV, "/n/gnb.crt"),
                (SBI_TLS_CLIENT_KEY_ENV, "/s/c.key"),
            ],
        );
        assert_eq!(t.client_key, None);
        assert!(t
            .error
            .as_deref()
            .unwrap()
            .contains("without its private key"));
        let t = resolve("https://nrf", &[(SBI_TLS_CLIENT_KEY_ENV, "/s/c.key")]);
        assert!(t
            .error
            .as_deref()
            .unwrap()
            .contains("without its certificate"));
        assert!(t
            .client_config()
            .unwrap_err()
            .contains("without its certificate"));
    }

    #[test]
    fn the_server_name_applies_to_the_nrf_authority_only_1617() {
        let t = NrfTls::with_ca("/ca").nrf_server_name("https://127.0.0.1:7777", "nrf");
        assert_eq!(t.server_name_for("127.0.0.1:7777", "127.0.0.1"), "nrf");
        assert_eq!(
            t.server_name_for("127.0.0.1:7778", "127.0.0.1"),
            "127.0.0.1"
        );
        assert_eq!(t.server_name_for("10.0.0.1:7777", "10.0.0.1"), "10.0.0.1");
        assert_eq!(
            NrfTls::with_ca("/ca").server_name_for("127.0.0.1:7777", "127.0.0.1"),
            "127.0.0.1"
        );
    }

    #[test]
    fn https_detection_1617() {
        assert!(is_https("https://nrf:7777"));
        assert!(is_https(" HTTPS://nrf"));
        assert!(!is_https("http://nrf:7777"));
        assert!(!is_https("nrf"));
    }
}
