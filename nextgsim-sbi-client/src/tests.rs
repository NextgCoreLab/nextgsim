//! Against an in-process h2c stub bound to 127.0.0.1:0.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use p256::ecdsa::signature::Verifier as _;
use tokio::net::TcpListener;

use super::*;

#[derive(Debug, Clone)]
struct Seen {
    method: Method,
    path: String,
    headers: HeaderMap,
    body: Bytes,
}

#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: String,
    delay: Duration,
}

impl Reply {
    fn new(status: u16, body: &str) -> Self {
        Self {
            status,
            headers: vec![],
            body: body.to_string(),
            delay: Duration::ZERO,
        }
    }
    fn header(mut self, k: &'static str, v: &str) -> Self {
        self.headers.push((k, v.to_string()));
        self
    }
    fn delayed(mut self, d: Duration) -> Self {
        self.delay = d;
        self
    }
}

type Handler = Arc<dyn Fn(&Seen) -> Reply + Send + Sync>;

struct Stub {
    addr: std::net::SocketAddr,
    seen: Arc<StdMutex<Vec<Seen>>>,
    /// Connections that have ended, and a wake-up when one does.
    closed: Arc<(AtomicUsize, tokio::sync::Notify)>,
}

impl Stub {
    async fn start(handler: impl Fn(&Seen) -> Reply + Send + Sync + 'static) -> Self {
        Self::start_with(false, handler).await
    }

    /// `one_request_per_conn`: each connection is shut down (GOAWAY) after its
    /// first request, as a peer closing an idle pooled connection would.
    async fn start_with(
        one_request_per_conn: bool,
        handler: impl Fn(&Seen) -> Reply + Send + Sync + 'static,
    ) -> Self {
        let closed = Arc::new((AtomicUsize::new(0), tokio::sync::Notify::new()));
        let closed2 = closed.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let handler: Handler = Arc::new(handler);
        let seen2 = seen.clone();
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let handler = handler.clone();
                let seen = seen2.clone();
                let closed = closed2.clone();
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(sock).await else {
                        return;
                    };
                    while let Some(Ok((req, mut respond))) = conn.accept().await {
                        if one_request_per_conn {
                            conn.graceful_shutdown();
                        }
                        let handler = handler.clone();
                        let seen = seen.clone();
                        tokio::spawn(async move {
                            let (parts, mut body) = req.into_parts();
                            let mut buf = BytesMut::new();
                            while let Some(Ok(c)) = body.data().await {
                                let _ = body.flow_control().release_capacity(c.len());
                                buf.extend_from_slice(&c);
                            }
                            let s = Seen {
                                method: parts.method,
                                path: parts
                                    .uri
                                    .path_and_query()
                                    .map_or_else(String::new, |p| p.as_str().to_string()),
                                headers: parts.headers,
                                body: buf.freeze(),
                            };
                            seen.lock().unwrap().push(s.clone());
                            let r = handler(&s);
                            tokio::time::sleep(r.delay).await;
                            let mut resp = http::Response::builder().status(r.status);
                            for (k, v) in &r.headers {
                                resp = resp.header(*k, v);
                            }
                            let Ok(mut send) = respond.send_response(resp.body(()).unwrap(), false)
                            else {
                                return;
                            };
                            let _ = send.send_data(Bytes::from(r.body), true);
                        });
                    }
                    closed.0.fetch_add(1, Ordering::SeqCst);
                    closed.1.notify_waiters();
                });
            }
        });
        Self { addr, seen, closed }
    }

    /// Wait until `n` connections have ended.
    async fn closed_at_least(&self, n: usize) {
        loop {
            let notified = self.closed.1.notified();
            if self.closed.0.load(Ordering::SeqCst) >= n {
                return;
            }
            notified.await;
        }
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }

    fn count(&self, path_prefix: &str) -> usize {
        self.seen()
            .iter()
            .filter(|s| s.path.starts_with(path_prefix))
            .count()
    }
}

fn fast() -> SbiClientConfig {
    SbiClientConfig {
        timeout: Duration::from_millis(300),
        retries: 2,
        retry_backoff: Duration::from_millis(10),
        ..Default::default()
    }
}

fn nwdaf() -> Scope {
    Scope::new("NWDAF", "nnwdaf-analyticsinfo")
}

// ---- transport, JSON, statuses ------------------------------------------

#[tokio::test]
async fn a_2xx_get_decodes_json_over_h2c() {
    let stub = Stub::start(|_| {
        Reply::new(200, r#"{"nfLoadLevelInfos":[{"nfLoadLevelAverage":7}]}"#)
            .header("content-type", "application/json")
    })
    .await;
    let c = SbiClient::new(fast());
    let v: serde_json::Value = c
        .get_json(
            &stub.url("/nnwdaf-analyticsinfo/v1/analytics?event-id=NF_LOAD"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(v["nfLoadLevelInfos"][0]["nfLoadLevelAverage"], 7);
    let seen = stub.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].method, Method::GET);
    assert_eq!(
        seen[0].path,
        "/nnwdaf-analyticsinfo/v1/analytics?event-id=NF_LOAD"
    );
    assert!(seen[0].headers.get(header::AUTHORIZATION).is_none());
}

#[tokio::test]
async fn a_post_sends_its_json_body_once_and_returns_the_201() {
    let stub = Stub::start(|_| Reply::new(201, "{}").header("location", "/subs/1")).await;
    let c = SbiClient::new(fast());
    let r = c
        .post_json(
            &stub.url("/nnwdaf-eventssubscription/v1/subscriptions"),
            &serde_json::json!({"eventSubscriptions":[{"event":"NF_LOAD"}]}),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r.status, StatusCode::CREATED);
    assert_eq!(r.headers["location"], "/subs/1");
    let seen = stub.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].headers[header::CONTENT_TYPE], "application/json");
    let body: serde_json::Value = serde_json::from_slice(&seen[0].body).unwrap();
    assert_eq!(body["eventSubscriptions"][0]["event"], "NF_LOAD");
}

#[tokio::test]
async fn a_large_body_is_sent_whole() {
    let stub = Stub::start(|_| Reply::new(204, "")).await;
    let c = SbiClient::new(fast());
    let big = "x".repeat(200_000); // beyond the 64 KiB initial window
    c.put_json(&stub.url("/big"), &big, None).await.unwrap();
    assert_eq!(stub.seen()[0].body.len(), big.len() + 2);
}

#[tokio::test]
async fn a_401_is_unauthorized_with_its_challenge_and_is_not_retried() {
    let stub = Stub::start(|_| {
        Reply::new(401, "{}").header(
            "www-authenticate",
            r#"Bearer realm="5gc-sbi", error="invalid_token""#,
        )
    })
    .await;
    let c = SbiClient::new(fast());
    let err = c
        .get_json::<serde_json::Value>(&stub.url("/a"), Some(&nwdaf()))
        .await
        .unwrap_err();
    match err {
        SbiError::Unauthorized { www_authenticate } => {
            assert!(www_authenticate
                .unwrap()
                .starts_with("Bearer realm=\"5gc-sbi\""));
        }
        other => panic!("expected Unauthorized, got {other:?}"),
    }
    assert_eq!(stub.count("/a"), 1, "a 401 is never retried");
}

#[tokio::test]
async fn a_403_is_forbidden() {
    let stub = Stub::start(|_| Reply::new(403, "{}")).await;
    let c = SbiClient::new(fast());
    assert!(matches!(
        c.get_json::<serde_json::Value>(&stub.url("/a"), None).await,
        Err(SbiError::Forbidden)
    ));
    assert_eq!(stub.count("/a"), 1);
}

#[tokio::test]
async fn another_non_2xx_is_a_status_error_for_the_json_helper() {
    let stub = Stub::start(|_| Reply::new(404, "{}")).await;
    let c = SbiClient::new(fast());
    assert!(matches!(
        c.get_json::<serde_json::Value>(&stub.url("/a"), None).await,
        Err(SbiError::Status { status: 404 })
    ));
}

// ---- timeouts and retries -----------------------------------------------

#[tokio::test]
async fn a_timed_out_get_is_retried_then_reported() {
    let stub = Stub::start(|_| Reply::new(200, "{}").delayed(Duration::from_secs(2))).await;
    let c = SbiClient::new(fast());
    let err = c
        .get_json::<serde_json::Value>(&stub.url("/slow"), None)
        .await
        .unwrap_err();
    assert!(matches!(err, SbiError::Timeout(_)), "{err:?}");
    assert_eq!(stub.count("/slow"), 3, "1 attempt + 2 retries");
}

#[tokio::test]
async fn a_timed_out_post_is_sent_exactly_once() {
    let stub = Stub::start(|_| Reply::new(201, "{}").delayed(Duration::from_secs(2))).await;
    let c = SbiClient::new(fast());
    let err = c
        .post_json(&stub.url("/subscriptions"), &serde_json::json!({}), None)
        .await
        .unwrap_err();
    assert!(matches!(err, SbiError::Timeout(_)), "{err:?}");
    // Wait past the client's would-be retries before counting.
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(stub.count("/subscriptions"), 1, "a POST is never retried");
}

#[tokio::test]
async fn a_503_get_is_retried_until_it_succeeds_but_a_503_post_is_not() {
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = n.clone();
    let stub = Stub::start(move |s| {
        if s.method == Method::GET && n2.fetch_add(1, Ordering::SeqCst) < 2 {
            Reply::new(503, "{}")
        } else if s.method == Method::GET {
            Reply::new(200, r#"{"ok":true}"#)
        } else {
            Reply::new(503, "{}")
        }
    })
    .await;
    let c = SbiClient::new(fast());
    let v: serde_json::Value = c.get_json(&stub.url("/g"), None).await.unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(stub.count("/g"), 3);

    let r = c
        .post_json(&stub.url("/p"), &serde_json::json!({}), None)
        .await
        .unwrap();
    assert_eq!(r.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(stub.count("/p"), 1, "a POST is never retried");
}

#[tokio::test]
async fn a_refused_connection_is_a_connect_error() {
    // Bind and keep the listener's port reserved, then close it: nothing
    // accepts there.
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    drop(l);
    let c = SbiClient::new(SbiClientConfig {
        retries: 0,
        ..fast()
    });
    let err = c
        .get_json::<serde_json::Value>(&format!("http://127.0.0.1:{port}/x"), None)
        .await
        .unwrap_err();
    assert!(matches!(err, SbiError::Connect { .. }), "{err:?}");
}

#[tokio::test]
async fn https_and_bad_uris_are_refused_before_anything_is_sent() {
    let c = SbiClient::new(fast());
    assert!(matches!(
        c.get_json::<serde_json::Value>("https://127.0.0.1:1/x", None)
            .await,
        Err(SbiError::UnsupportedScheme(s)) if s == "https"
    ));
    assert!(matches!(
        c.get_json::<serde_json::Value>("/relative", None).await,
        Err(SbiError::InvalidUri(_))
    ));
}

#[test]
fn only_rfc9110_idempotent_methods_are_retryable() {
    for m in [
        Method::GET,
        Method::HEAD,
        Method::PUT,
        Method::DELETE,
        Method::OPTIONS,
    ] {
        assert!(is_idempotent(&m), "{m}");
    }
    for m in [Method::POST, Method::PATCH] {
        assert!(!is_idempotent(&m), "{m}");
    }
}

// ---- OAuth2 ---------------------------------------------------------------

const ID: &str = "6b1d7e3c-0000-4000-8000-00000000457a";
const TOKEN: &str = "eyJhbGciOiJFUzI1NiJ9.the-secret-token.sig";

fn form(body: &Bytes) -> HashMap<String, String> {
    std::str::from_utf8(body)
        .unwrap()
        .split('&')
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            let dec = |s: &str| {
                let s = s.replace('+', " ");
                let mut out = Vec::new();
                let b = s.as_bytes();
                let mut i = 0;
                while i < b.len() {
                    if b[i] == b'%' {
                        out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap());
                        i += 3;
                    } else {
                        out.push(b[i]);
                        i += 1;
                    }
                }
                String::from_utf8(out).unwrap()
            };
            (dec(k), dec(v))
        })
        .collect()
}

/// An NRF + producer stub: the token endpoint answers `token_reply`, every other
/// path answers 200 when it sees `Bearer TOKEN`, else 401.
async fn nrf_and_producer(token_reply: Reply) -> Stub {
    Stub::start(move |s| {
        if s.path == TOKEN_PATH {
            token_reply.clone()
        } else if s
            .headers
            .get(header::AUTHORIZATION)
            .is_some_and(|v| v == format!("Bearer {TOKEN}").as_str())
        {
            Reply::new(200, r#"{"ok":true}"#)
        } else {
            Reply::new(401, "{}").header("www-authenticate", r#"Bearer error="invalid_token""#)
        }
    })
    .await
}

fn granted() -> Reply {
    Reply::new(
        200,
        &format!(
            r#"{{"access_token":"{TOKEN}","token_type":"Bearer","expires_in":3600,"scope":"nnwdaf-analyticsinfo"}}"#
        ),
    )
}

fn oauth_client(stub: &Stub, key: &CcaKey) -> SbiClient {
    SbiClient::new(SbiClientConfig {
        oauth2: Some(OAuth2Config::new(
            format!("http://{}", stub.addr),
            ID,
            "AF",
            key.clone(),
        )),
        ..fast()
    })
}

#[tokio::test]
async fn the_token_request_matches_nextgcores_nrf_and_the_token_is_attached_and_cached() {
    let stub = nrf_and_producer(granted()).await;
    let key = CcaKey::generate();
    let c = oauth_client(&stub, &key);
    for _ in 0..2 {
        let v: serde_json::Value = c
            .get_json(&stub.url("/nnwdaf-x"), Some(&nwdaf()))
            .await
            .unwrap();
        assert_eq!(v["ok"], true);
    }
    let seen = stub.seen();
    let token_reqs: Vec<_> = seen.iter().filter(|s| s.path == TOKEN_PATH).collect();
    assert_eq!(token_reqs.len(), 1, "the token is cached across calls");
    let t = token_reqs[0];
    assert_eq!(t.method, Method::POST);
    assert_eq!(
        t.headers[header::CONTENT_TYPE],
        "application/x-www-form-urlencoded"
    );
    let f = form(&t.body);
    assert_eq!(f["grant_type"], "client_credentials");
    assert_eq!(f["nfInstanceId"], ID);
    assert_eq!(f["nfType"], "AF");
    assert_eq!(f["targetNfType"], "NWDAF");
    assert_eq!(f["scope"], "nnwdaf-analyticsinfo");

    // The CCA: ES256 header, the claims nextgcore's NRF checks, and a
    // signature that verifies under the published key.
    let parts: Vec<&str> = f["cca"].split('.').collect();
    assert_eq!(parts.len(), 3);
    let hdr: serde_json::Value =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[0]).unwrap()).unwrap();
    assert_eq!(hdr, serde_json::json!({"alg":"ES256","typ":"JWT"}));
    let claims: serde_json::Value =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[1]).unwrap()).unwrap();
    assert_eq!(claims["sub"], ID);
    assert_eq!(claims["iss"], ID);
    assert_eq!(claims["aud"], "NRF");
    let iat = claims["iat"].as_u64().unwrap();
    assert_eq!(claims["exp"].as_u64().unwrap(), iat + 60);
    assert!(iat.abs_diff(now_for_test()) < 5);
    let sig = Signature::from_slice(&URL_SAFE_NO_PAD.decode(parts[2]).unwrap()).unwrap();
    key.verifying_key()
        .verify(format!("{}.{}", parts[0], parts[1]).as_bytes(), &sig)
        .expect("the CCA verifies under the requester's key");

    let producer: Vec<_> = seen.iter().filter(|s| s.path == "/nnwdaf-x").collect();
    assert_eq!(producer.len(), 2);
    for p in producer {
        assert_eq!(
            p.headers[header::AUTHORIZATION],
            format!("Bearer {TOKEN}").as_str()
        );
    }
}

use p256::ecdsa::Signature;

fn now_for_test() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[tokio::test]
async fn a_refused_token_fails_closed_and_no_tokenless_call_is_made() {
    let stub = nrf_and_producer(Reply::new(
        400,
        r#"{"error":"invalid_client","error_description":"NF instance is not registered"}"#,
    ))
    .await;
    let c = oauth_client(&stub, &CcaKey::generate());
    let err = c
        .get_json::<serde_json::Value>(&stub.url("/nnwdaf-x"), Some(&nwdaf()))
        .await
        .unwrap_err();
    match err {
        SbiError::TokenRefused { status, error } => {
            assert_eq!(status, 400);
            assert_eq!(error.as_deref(), Some("invalid_client"));
        }
        other => panic!("expected TokenRefused, got {other:?}"),
    }
    assert_eq!(stub.count("/nnwdaf-x"), 0, "no call without a token");
    assert_eq!(stub.count(TOKEN_PATH), 1, "the token POST is not retried");
}

#[tokio::test]
async fn an_unreachable_nrf_fails_closed() {
    let producer = nrf_and_producer(granted()).await;
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let dead = l.local_addr().unwrap();
    drop(l);
    let c = SbiClient::new(SbiClientConfig {
        oauth2: Some(OAuth2Config::new(
            format!("http://{dead}"),
            ID,
            "AF",
            CcaKey::generate(),
        )),
        ..fast()
    });
    assert!(matches!(
        c.get_json::<serde_json::Value>(&producer.url("/nnwdaf-x"), Some(&nwdaf()))
            .await,
        Err(SbiError::Connect { .. })
    ));
    assert_eq!(producer.count("/nnwdaf-x"), 0);
}

#[tokio::test]
async fn with_oauth2_a_call_that_names_no_scope_is_refused_unsent() {
    let stub = nrf_and_producer(granted()).await;
    let c = oauth_client(&stub, &CcaKey::generate());
    assert!(matches!(
        c.get_json::<serde_json::Value>(&stub.url("/nnwdaf-x"), None)
            .await,
        Err(SbiError::ScopeRequired)
    ));
    assert!(stub.seen().is_empty());
}

#[tokio::test]
async fn a_non_bearer_or_empty_token_is_rejected() {
    for body in [
        r#"{"access_token":"t","token_type":"mac","expires_in":60}"#,
        r#"{"access_token":"","token_type":"Bearer"}"#,
        r#"{"nope":1}"#,
    ] {
        let stub = nrf_and_producer(Reply::new(200, body)).await;
        let c = oauth_client(&stub, &CcaKey::generate());
        let err = c
            .get_json::<serde_json::Value>(&stub.url("/nnwdaf-x"), Some(&nwdaf()))
            .await
            .unwrap_err();
        assert!(matches!(err, SbiError::TokenInvalid(_)), "{body}: {err:?}");
        assert_eq!(stub.count("/nnwdaf-x"), 0);
    }
}

#[tokio::test]
async fn a_401_with_a_token_drops_it_so_the_next_call_fetches_a_fresh_one() {
    let reject_first = Arc::new(AtomicUsize::new(0));
    let r2 = reject_first.clone();
    let stub = Stub::start(move |s| {
        if s.path == TOKEN_PATH {
            granted()
        } else if r2.fetch_add(1, Ordering::SeqCst) == 0 {
            Reply::new(401, "{}")
        } else {
            Reply::new(200, "{}")
        }
    })
    .await;
    let c = oauth_client(&stub, &CcaKey::generate());
    assert!(matches!(
        c.get_json::<serde_json::Value>(&stub.url("/p"), Some(&nwdaf()))
            .await,
        Err(SbiError::Unauthorized { .. })
    ));
    c.get_json::<serde_json::Value>(&stub.url("/p"), Some(&nwdaf()))
        .await
        .unwrap();
    assert_eq!(stub.count(TOKEN_PATH), 2);
}

/// Every test in this binary logs into one global capturing subscriber, so the
/// no-secrets check below sees all of their output, not one thread's.
static LOGS: std::sync::OnceLock<Arc<StdMutex<Vec<u8>>>> = std::sync::OnceLock::new();

fn global_logs() -> Arc<StdMutex<Vec<u8>>> {
    LOGS.get_or_init(|| {
        use std::io::Write;
        #[derive(Clone)]
        struct Cap(Arc<StdMutex<Vec<u8>>>);
        impl Write for Cap {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buf = Arc::new(StdMutex::new(Vec::new()));
        let w = Cap(buf.clone());
        tracing_subscriber::fmt()
            .with_writer(move || w.clone())
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .init();
        buf
    })
    .clone()
}

#[tokio::test]
async fn no_token_or_key_reaches_the_logs_or_debug_output() {
    let cap = global_logs();
    let stub = nrf_and_producer(granted()).await;
    let key = CcaKey::generate();
    let c = oauth_client(&stub, &key);
    c.get_json::<serde_json::Value>(&stub.url("/nnwdaf-x"), Some(&nwdaf()))
        .await
        .unwrap();
    let logs = String::from_utf8(cap.lock().unwrap().clone()).unwrap();
    assert!(
        logs.contains("NRF granted a token"),
        "the path was logged: {logs}"
    );
    let scalar = hex::encode(key.0.to_bytes());
    for secret in [TOKEN, "the-secret-token", scalar.as_str()] {
        assert!(!logs.contains(secret), "secret in logs");
        assert!(!format!("{c:?}").contains(secret), "secret in client Debug");
    }
    assert_eq!(format!("{key:?}"), "CcaKey(<redacted>)");
    assert_eq!(
        format!("{:?}", AccessToken(TOKEN.into())),
        "AccessToken(<redacted>)"
    );
}

// ---- key files ------------------------------------------------------------

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "sbi-client-{tag}-{}-{}",
        std::process::id(),
        now_for_test()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn a_key_file_is_created_0600_reused_and_never_silently_replaced() {
    let d = tmpdir("key");
    let p = d.join("cca.key");
    let k1 = CcaKey::load_or_create(&p).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&p).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    let k2 = CcaKey::load_or_create(&p).unwrap();
    assert_eq!(k1.verifying_key(), k2.verifying_key(), "the file is reused");
    std::fs::write(&p, "not-hex").unwrap();
    assert!(matches!(CcaKey::load_or_create(&p), Err(SbiError::Key(_))));
    assert_eq!(
        std::fs::read_to_string(&p).unwrap(),
        "not-hex",
        "never regenerated"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_published_jwk_is_the_nrf_trust_store_shape() {
    let d = tmpdir("jwk");
    let key = CcaKey::generate();
    let path = key.publish_jwk(&d, ID).unwrap();
    assert_eq!(path, d.join(format!("{ID}.jwk")));
    let jwk: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(jwk["kty"], "EC");
    assert_eq!(jwk["crv"], "P-256");
    assert_eq!(jwk["alg"], "ES256");
    assert_eq!(jwk["use"], "sig");
    assert_eq!(jwk["kid"], ID);
    let x = URL_SAFE_NO_PAD.decode(jwk["x"].as_str().unwrap()).unwrap();
    let y = URL_SAFE_NO_PAD.decode(jwk["y"].as_str().unwrap()).unwrap();
    let mut sec1 = vec![4u8];
    sec1.extend(x);
    sec1.extend(y);
    assert_eq!(
        p256::ecdsa::VerifyingKey::from_sec1_bytes(&sec1).unwrap(),
        key.verifying_key()
    );
    assert!(key.publish_jwk(&d, "../escape").is_err());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn form_encoding_escapes_reserved_bytes_and_uses_plus_for_space() {
    assert_eq!(
        oauth::form_encode(&[("scope", "a b"), ("x", "=&+/é")]),
        "scope=a+b&x=%3D%26%2B%2F%C3%A9"
    );
}

// ---- review fixes: isolation, expiry overflow, a closed pooled connection ----

/// A listener whose accept queue is full and never drained: the kernel drops
/// further SYNs, so a connect to it hangs, like a blackholed producer.
fn blackhole() -> (
    socket2::Socket,
    Vec<std::net::TcpStream>,
    std::net::SocketAddr,
) {
    let l = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, None).unwrap();
    l.bind(
        &"127.0.0.1:0"
            .parse::<std::net::SocketAddr>()
            .unwrap()
            .into(),
    )
    .unwrap();
    l.listen(0).unwrap();
    let addr = l.local_addr().unwrap().as_socket().unwrap();
    let mut fill = Vec::new();
    for _ in 0..8 {
        if let Ok(s) = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(100)) {
            fill.push(s);
        } else {
            break;
        }
    }
    (l, fill, addr)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hanging_connect_to_one_producer_does_not_delay_another() {
    let (_l, _fill, dead) = blackhole();
    let healthy = Stub::start(|_| Reply::new(201, "{}")).await;
    let c = Arc::new(SbiClient::new(SbiClientConfig {
        timeout: Duration::from_millis(1500),
        retries: 0,
        ..fast()
    }));
    // Precondition: the blackhole really hangs a connect.
    let c2 = c.clone();
    let stuck = tokio::spawn(async move {
        c2.get_json::<serde_json::Value>(&format!("http://{dead}/x"), None)
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let t0 = std::time::Instant::now();
    let r = c
        .post_json(&healthy.url("/subscriptions"), &serde_json::json!({}), None)
        .await
        .expect("the healthy producer answers");
    let took = t0.elapsed();
    assert_eq!(r.status, StatusCode::CREATED);
    assert!(
        took < Duration::from_millis(500),
        "the healthy POST waited {took:?} behind the hanging connect"
    );
    assert!(matches!(stuck.await.unwrap(), Err(SbiError::Timeout(_))));
}

#[tokio::test]
async fn a_huge_expires_in_is_clamped_not_a_panic() {
    let stub = nrf_and_producer(Reply::new(
        200,
        &format!(r#"{{"access_token":"{TOKEN}","token_type":"Bearer","expires_in":18446744073709551615}}"#),
    ))
    .await;
    let c = oauth_client(&stub, &CcaKey::generate());
    for _ in 0..2 {
        c.get_json::<serde_json::Value>(&stub.url("/nnwdaf-x"), Some(&nwdaf()))
            .await
            .unwrap();
    }
    assert_eq!(stub.count(TOKEN_PATH), 1, "cached, not re-fetched");
    let now = std::time::Instant::now();
    assert!(oauth::refresh_at(now, u64::MAX) <= now + Duration::from_secs(86_400));
    // A lifetime inside the refresh margin is not cached at all.
    assert_eq!(oauth::refresh_at(now, 10), now);
}

#[tokio::test]
async fn a_pooled_connection_the_peer_closed_is_replaced_and_a_post_still_goes_once() {
    let stub = Stub::start_with(true, |_| Reply::new(201, "{}")).await;
    let c = SbiClient::new(SbiClientConfig {
        retries: 0,
        ..fast()
    });
    c.get_json::<serde_json::Value>(&stub.url("/first"), None)
        .await
        .unwrap();
    // The peer shut that connection down after its one request.
    stub.closed_at_least(1).await;
    let r = c
        .post_json(&stub.url("/subscriptions"), &serde_json::json!({}), None)
        .await
        .expect("a POST on a closed pooled connection reconnects; nothing was sent yet");
    assert_eq!(r.status, StatusCode::CREATED);
    assert_eq!(stub.count("/subscriptions"), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_slow_token_fetch_for_one_scope_does_not_delay_another() {
    let stub = Stub::start(|s| {
        if s.path == TOKEN_PATH {
            let slow = std::str::from_utf8(&s.body).unwrap().contains("scope=slow");
            granted().delayed(if slow {
                Duration::from_millis(800)
            } else {
                Duration::ZERO
            })
        } else {
            Reply::new(200, "{}")
        }
    })
    .await;
    let c = Arc::new(SbiClient::new(SbiClientConfig {
        timeout: Duration::from_secs(2),
        ..oauth_client(&stub, &CcaKey::generate()).config().clone()
    }));
    let c2 = c.clone();
    let url = stub.url("/p");
    let slow = tokio::spawn(async move {
        c2.get_json::<serde_json::Value>(&url, Some(&Scope::new("NWDAF", "slow")))
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let t0 = std::time::Instant::now();
    c.get_json::<serde_json::Value>(&stub.url("/p"), Some(&nwdaf()))
        .await
        .unwrap();
    assert!(
        t0.elapsed() < Duration::from_millis(500),
        "{:?}",
        t0.elapsed()
    );
    slow.await.unwrap().unwrap();
}
