//! `sbi-roundtrip`: the nightly's round trip through a real nextgcore NRF and
//! nwdafd with `nextgsim-sbi-client` (#457). Built only with `--features
//! roundtrip`.
//!
//! Steps, each printed as one `ROUNDTRIP <step> ...` line so the job can assert
//! the order:
//! 1. `nrf-register`: PUT this client's NF profile (nfType `NF_TYPE`) to the NRF.
//! 2. `nrf-discover`: discover NWDAF through the NRF and take its SBI endpoint.
//! 3. In `oauth2` mode only:
//!    - `tokenless`: a GET with no token must be refused by nwdafd with 401;
//!    - `token`: with the CCA key published to the NRF's trusted-keys directory,
//!      the NRF must grant a token for `nnwdaf-analyticsinfo`.
//! 4. `nwdaf-analytics`: `GET /nnwdaf-analyticsinfo/v1/analytics?event-id=NF_LOAD`
//!    must answer 2xx.
//! 5. `nrf-deregister`, then `OK`.
//!
//! Environment:
//! - `NRF_URI` (default `http://172.23.0.10:7777`);
//! - `MODE` = `plain` (default) or `oauth2`;
//! - `NWDAF_URI`, used when discovery returns no NWDAF;
//! - `NF_TYPE` (default `AF`) and `NF_INSTANCE_ID` (default a fixed dev UUID);
//! - `CCA_KEY_FILE`, `CCA_PUBLIC_JWK_DIR` (oauth2 mode);
//! - `EXPECT_TOKEN` = `granted` (default) or `refused:<error>` (oauth2 mode): with
//!   `refused:`, the NRF's refusal with that `error` is the expected result.
//!
//! Exit 0 on success, 1 with a `ROUNDTRIP FAIL <step>: <why>` line otherwise.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use nextgsim_sbi_client::{CcaKey, OAuth2Config, SbiClient, SbiClientConfig, SbiError, Scope};
use serde_json::{json, Value};

const DEFAULT_ID: &str = "6b1d7e3c-0000-4000-8000-00000000457a";

struct Fail(&'static str, String);

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn nwdaf_endpoint(discovery: &Value) -> Option<String> {
    let insts = discovery.get("nfInstances")?.as_array()?;
    for inst in insts {
        for svc in inst
            .get("nfServices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            for ep in svc
                .get("ipEndPoints")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let (Some(ip), Some(port)) = (
                    ep.get("ipv4Address").and_then(Value::as_str),
                    ep.get("port").and_then(Value::as_u64),
                ) {
                    return Some(format!("http://{ip}:{port}"));
                }
            }
        }
        if let Some(ip) = inst
            .get("ipv4Addresses")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
        {
            return Some(format!("http://{ip}:7777"));
        }
    }
    None
}

async fn run() -> Result<(), Fail> {
    let nrf = env("NRF_URI", "http://172.23.0.10:7777");
    let mode = env("MODE", "plain");
    let id = env("NF_INSTANCE_ID", DEFAULT_ID);
    // AF: TS 29.520 §4.3.1.3.2 lists the AF as an Nnwdaf_AnalyticsInfo consumer.
    let nf_type = env("NF_TYPE", "AF");
    let base = SbiClientConfig {
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    // NRF management is not token-protected in nextgcore's oauth2 profile
    // (nrf.sbi.oauth2.require_server is unset), so registration is tokenless by
    // design, on a client with no OAuth2 configured.
    let plain = SbiClient::new(base.clone());

    let profile = json!({
        "nfInstanceId": id,
        "nfType": nf_type,
        "nfStatus": "REGISTERED",
        "heartBeatTimer": 60,
    });
    let nf_uri = format!("{nrf}/nnrf-nfm/v1/nf-instances/{id}");
    let r = plain
        .put_json(&nf_uri, &profile, None)
        .await
        .map_err(|e| Fail("nrf-register", e.to_string()))?;
    if !r.status.is_success() {
        return Err(Fail("nrf-register", format!("status {}", r.status)));
    }
    println!("ROUNDTRIP nrf-register status={}", r.status.as_u16());

    let disc: Value = plain
        .get_json(
            &format!(
                "{nrf}/nnrf-disc/v1/nf-instances?target-nf-type=NWDAF&requester-nf-type={nf_type}"
            ),
            None,
        )
        .await
        .map_err(|e| Fail("nrf-discover", e.to_string()))?;
    let found = disc
        .get("nfInstances")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    // NWDAF_URI is an optional fallback for a core whose NWDAF profile leaves
    // the requester type out of allowedNfTypes (nextgcore before #1412 did so
    // for AF). The nightly passes none, so there discovery must find the NWDAF.
    let nwdaf = match (nwdaf_endpoint(&disc), std::env::var("NWDAF_URI")) {
        (Some(ep), _) => ep,
        (None, Ok(uri)) => uri,
        (None, Err(_)) => {
            return Err(Fail(
                "nrf-discover",
                format!("no NWDAF endpoint in {disc} and no NWDAF_URI"),
            ))
        }
    };
    println!("ROUNDTRIP nrf-discover status=200 nwdaf-instances={found} nwdaf={nwdaf}");

    let scope = Scope::new("NWDAF", "nnwdaf-analyticsinfo");
    let analytics = format!("{nwdaf}/nnwdaf-analyticsinfo/v1/analytics?event-id=NF_LOAD");
    let client = match mode.as_str() {
        "plain" => plain,
        "oauth2" => {
            match plain
                .request(http::Method::GET, &analytics, None, Some(&scope))
                .await
            {
                Err(SbiError::Unauthorized { www_authenticate }) => println!(
                    "ROUNDTRIP tokenless status=401 (expected) www-authenticate={}",
                    www_authenticate.as_deref().unwrap_or("-")
                ),
                other => {
                    return Err(Fail(
                        "tokenless",
                        format!("nwdafd must refuse a tokenless call with 401, got {other:?}"),
                    ))
                }
            }
            let key_file = PathBuf::from(env("CCA_KEY_FILE", "/tmp/sbi-roundtrip-cca.key"));
            let key =
                CcaKey::load_or_create(&key_file).map_err(|e| Fail("token", e.to_string()))?;
            if let Ok(dir) = std::env::var("CCA_PUBLIC_JWK_DIR") {
                key.publish_jwk(&PathBuf::from(dir), &id)
                    .map_err(|e| Fail("token", e.to_string()))?;
            }
            SbiClient::new(SbiClientConfig {
                oauth2: Some(OAuth2Config::new(
                    nrf.clone(),
                    id.clone(),
                    nf_type.clone(),
                    key,
                )),
                ..base
            })
        }
        other => {
            return Err(Fail(
                "mode",
                format!("MODE must be plain or oauth2, got {other:?}"),
            ))
        }
    };

    let expect = env("EXPECT_TOKEN", "granted");
    let r = match client
        .request(http::Method::GET, &analytics, None, Some(&scope))
        .await
    {
        Ok(r) => {
            if mode == "oauth2" {
                if expect != "granted" {
                    return Err(Fail(
                        "token",
                        format!("expected {expect}, but a token was granted"),
                    ));
                }
                println!("ROUNDTRIP token granted target=NWDAF scope=nnwdaf-analyticsinfo");
            }
            r
        }
        // EXPECT_TOKEN=refused:<error>: the NRF is expected to refuse (e.g.
        // unauthorized_client for a consumer type the NWDAF does not allow). The client must
        // fail closed: the refusal is the result and no tokenless call follows.
        Err(SbiError::TokenRefused { status, error })
            if mode == "oauth2" && expect.strip_prefix("refused:") == error.as_deref() =>
        {
            println!(
                "ROUNDTRIP token refused status={status} error={} (expected); failed closed, no \
                 tokenless call",
                error.as_deref().unwrap_or("-")
            );
            return deregister(&nf_uri, &mode).await;
        }
        Err(e) => return Err(Fail("nwdaf-analytics", e.to_string())),
    };
    if !r.status.is_success() {
        return Err(Fail(
            "nwdaf-analytics",
            format!("status {}: {}", r.status, String::from_utf8_lossy(&r.body)),
        ));
    }
    println!(
        "ROUNDTRIP nwdaf-analytics status={} bytes={}",
        r.status.as_u16(),
        r.body.len()
    );

    deregister(&nf_uri, &mode).await
}

async fn deregister(nf_uri: &str, mode: &str) -> Result<(), Fail> {
    let r = SbiClient::new(SbiClientConfig::default())
        .delete(nf_uri, None)
        .await
        .map_err(|e| Fail("nrf-deregister", e.to_string()))?;
    println!("ROUNDTRIP nrf-deregister status={}", r.status.as_u16());
    println!("ROUNDTRIP OK mode={mode}");
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(Fail(step, why)) => {
            println!("ROUNDTRIP FAIL {step}: {why}");
            ExitCode::FAILURE
        }
    }
}
