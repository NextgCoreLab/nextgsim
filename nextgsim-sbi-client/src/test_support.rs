//! Test support for `https://` NRF calls (nextgcore#1617), feature
//! `test-support`: a dev PKI, a TLS front with optional client auth (as
//! nextgcore's nrfd listener has it) that records the certificate each client
//! presented, and a guard over the `NEXTGCORE_NRF_TLS_*` variables.
//!
//! The front terminates TLS and relays the plaintext HTTP/2 to an existing
//! h2c stub, so a consumer's NRF stub serves `https://` unchanged.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair,
    KeyUsagePurpose, SanType,
};
use rustls::pki_types::pem::PemObject as _;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio::net::{TcpListener, TcpStream};

use crate::tls::{ENV_VARS, NRF_TLS_ENV_LOCK};

/// One issued certificate: its PEM files and DER.
#[derive(Debug, Clone)]
pub struct Issued {
    /// The certificate (PEM).
    pub cert: PathBuf,
    /// Its private key (PEM, 0600).
    pub key: PathBuf,
    /// The certificate's DER, to compare with what a server saw.
    pub der: Vec<u8>,
}

/// A CA issuing NF certificates the way nextgcore's `generate-dev-certs.sh`
/// does: `URI:urn:uuid:<nfInstanceId>` and the DNS / IP SANs given,
/// serverAuth + clientAuth.
pub struct TestPki {
    dir: PathBuf,
    ca: rcgen::Certificate,
    ca_key: KeyPair,
    /// The CA certificate (PEM).
    pub ca_cert: PathBuf,
}

impl TestPki {
    /// A new CA writing its files into `dir`.
    pub fn new(dir: &Path) -> Self {
        let mut params = CertificateParams::default();
        params
            .distinguished_name
            .push(DnType::CommonName, "nextgsim test CA");
        params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        let ca_key = KeyPair::generate().expect("CA key");
        let ca = params.self_signed(&ca_key).expect("CA certificate");
        let ca_cert = dir.join("ca.crt");
        std::fs::write(&ca_cert, ca.pem()).expect("write ca.crt");
        Self {
            dir: dir.to_path_buf(),
            ca,
            ca_key,
            ca_cert,
        }
    }

    /// Issue `<name>.crt` / `<name>.key` for `dns` and `ips`, with the URN of
    /// `nf_instance_id` if given.
    pub fn issue(
        &self,
        name: &str,
        dns: &[&str],
        ips: &[std::net::IpAddr],
        nf_instance_id: Option<&str>,
    ) -> Issued {
        let mut params =
            CertificateParams::new(dns.iter().map(ToString::to_string).collect::<Vec<_>>())
                .expect("certificate params");
        params.distinguished_name.push(DnType::CommonName, name);
        params
            .subject_alt_names
            .extend(ips.iter().map(|ip| SanType::IpAddress(*ip)));
        if let Some(id) = nf_instance_id {
            params.subject_alt_names.push(SanType::URI(
                format!("urn:uuid:{id}").try_into().expect("URN"),
            ));
        }
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        params.extended_key_usages = vec![
            ExtendedKeyUsagePurpose::ServerAuth,
            ExtendedKeyUsagePurpose::ClientAuth,
        ];
        let key = KeyPair::generate().expect("key");
        let cert = params
            .signed_by(&key, &self.ca, &self.ca_key)
            .expect("certificate");
        let (cert_path, key_path) = (
            self.dir.join(format!("{name}.crt")),
            self.dir.join(format!("{name}.key")),
        );
        std::fs::write(&cert_path, cert.pem()).expect("write certificate");
        std::fs::write(&key_path, key.serialize_pem()).expect("write key");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600));
        }
        Issued {
            cert: cert_path,
            key: key_path,
            der: cert.der().to_vec(),
        }
    }
}

/// What the front saw of one TLS connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Peer {
    /// The handshake completed; the client's certificate DER, if it sent one.
    Verified(Option<Vec<u8>>),
    /// The handshake failed.
    HandshakeFailed(String),
}

/// A TLS front on 127.0.0.1:0 relaying to `backend`: the server certificate
/// given, client certificates verified against `ca` but optional (rustls
/// `allow_unauthenticated`, as nrfd's `client_auth: optional`), ALPN `h2`.
pub struct TlsFront {
    /// Where it listens.
    pub addr: SocketAddr,
    peers: Arc<StdMutex<Vec<Peer>>>,
    accepted: Arc<AtomicUsize>,
}

impl TlsFront {
    /// Start the front for `backend` with `server` (its chain and key) and the
    /// client CA `ca`.
    pub async fn start(backend: SocketAddr, server: &Issued, ca: &Path) -> Self {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut roots = rustls::RootCertStore::empty();
        for c in CertificateDer::pem_file_iter(ca).expect("CA") {
            roots.add(c.expect("CA PEM")).expect("CA root");
        }
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            Arc::new(roots),
            provider.clone(),
        )
        .allow_unauthenticated()
        .build()
        .expect("client verifier");
        let chain = CertificateDer::pem_file_iter(&server.cert)
            .expect("server certificate")
            .collect::<Result<Vec<_>, _>>()
            .expect("server PEM");
        let key = PrivateKeyDer::from_pem_file(&server.key).expect("server key");
        let mut config = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .expect("versions")
            .with_client_cert_verifier(verifier)
            .with_single_cert(chain, key)
            .expect("server config");
        config.alpn_protocols = vec![b"h2".to_vec()];
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let peers = Arc::new(StdMutex::new(Vec::new()));
        let accepted = Arc::new(AtomicUsize::new(0));
        let (peers2, accepted2) = (peers.clone(), accepted.clone());
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                accepted2.fetch_add(1, Ordering::SeqCst);
                let (acceptor, peers) = (acceptor.clone(), peers2.clone());
                tokio::spawn(async move {
                    let mut tls = match acceptor.accept(sock).await {
                        Ok(t) => t,
                        Err(e) => {
                            peers
                                .lock()
                                .unwrap()
                                .push(Peer::HandshakeFailed(e.to_string()));
                            return;
                        }
                    };
                    let cert = tls
                        .get_ref()
                        .1
                        .peer_certificates()
                        .and_then(|c| c.first())
                        .map(|c| c.as_ref().to_vec());
                    peers.lock().unwrap().push(Peer::Verified(cert));
                    let Ok(mut up) = TcpStream::connect(backend).await else {
                        return;
                    };
                    let _ = tokio::io::copy_bidirectional(&mut tls, &mut up).await;
                });
            }
        });
        Self {
            addr,
            peers,
            accepted,
        }
    }

    /// `https://127.0.0.1:<port>`.
    pub fn url(&self) -> String {
        format!("https://{}", self.addr)
    }

    /// Every connection so far.
    pub fn peers(&self) -> Vec<Peer> {
        self.peers.lock().unwrap().clone()
    }

    /// TCP connections accepted so far (handshake or not).
    pub fn accepted(&self) -> usize {
        self.accepted.load(Ordering::SeqCst)
    }
}

/// Holds [`NRF_TLS_ENV_LOCK`] with the `NEXTGCORE_NRF_TLS_*` /
/// `NEXTGCORE_SBI_TLS_*` variables set to `vars` (every other one unset);
/// unsets them all on drop.
pub struct NrfTlsEnv {
    _guard: tokio::sync::MutexGuard<'static, ()>,
}

impl NrfTlsEnv {
    /// Take the lock and set `vars`.
    pub async fn set(vars: &[(&str, &Path)]) -> Self {
        let guard = NRF_TLS_ENV_LOCK.lock().await;
        clear();
        for (k, v) in vars {
            assert!(ENV_VARS.contains(k), "{k} is not an NRF TLS variable");
            std::env::set_var(k, v);
        }
        Self { _guard: guard }
    }
}

impl Drop for NrfTlsEnv {
    fn drop(&mut self) {
        clear();
    }
}

fn clear() {
    for k in ENV_VARS {
        std::env::remove_var(k);
    }
}
