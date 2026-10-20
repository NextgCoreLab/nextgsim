//! LTE UE configuration. Shape follows the NR UE's (`plmn`, a search list of
//! RLS addresses), so the two can sit side by side in one deployment.

use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;

use nextgsim_common::config::OpType;
use nextgsim_common::Plmn;
use nextgsim_eps_nas::aka::UsimCredentials;
use nextgsim_eps_nas::ie::PdnType;
use nextgsim_eps_nas::EpsNasConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The RLS port an eNB listens on when an entry gives none.
pub const DEFAULT_RLS_PORT: u16 = 4997;

/// LTE UE configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LteUeConfig {
    /// The PLMN NAS has selected (TS 36.304 §4.1: cell selection looks for a
    /// suitable cell of the selected PLMN).
    pub plmn: Plmn,
    /// eNB RLS endpoints to search, `"ip"` or `"ip:port"`.
    pub enb_search_list: Vec<String>,
    /// RLS sender identity. Random when absent.
    #[serde(default)]
    pub sti: Option<u64>,
    /// Simulated position the eNB derives the signal level from.
    #[serde(default)]
    pub position: [i32; 3],
    /// The USIM and what the UE attaches for. Without it the UE has no EPS
    /// NAS: it selects a cell and camps (TS 36.304 §5.2) and never attaches.
    #[serde(default)]
    pub subscriber: Option<SubscriberConfig>,
    /// The TUN interface the default bearer's PDN address is assigned to.
    /// Absent, no interface is created and IP packets only flow through the
    /// [`crate::task::IpIo`] seam (tests).
    #[serde(default)]
    pub tun: Option<TunSettings>,
}

/// The UE's TUN interface (as the NR UE's, `nextgsim_ue::tun`). Creating one
/// needs `CAP_NET_ADMIN`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TunSettings {
    /// Interface name; `lteuetun<EBI>` when absent.
    #[serde(default)]
    pub name: Option<String>,
    /// Whether to install a route for the PDN address's traffic.
    #[serde(default)]
    pub configure_routing: bool,
}

/// The subscription (USIM) and the UE's EPS capabilities. Field names follow
/// the NR UE's (`key`, `op`, `op_type`, `amf`, `imei_sv`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscriberConfig {
    /// IMSI digits (TS 23.003 §2.2): MCC, MNC, MSIN.
    pub imsi: String,
    /// K.
    pub key: [u8; 16],
    /// OP or OPc, per `op_type`.
    pub op: [u8; 16],
    /// Whether `op` is OP or OPc.
    #[serde(default)]
    pub op_type: OpType,
    /// The USIM's highest accepted SQN at power-on (TS 33.102 Annex C).
    #[serde(default)]
    pub sqn: u64,
    /// IMEISV, 16 digits, sent when a SECURITY MODE COMMAND requests it.
    #[serde(default)]
    pub imei_sv: Option<String>,
    /// Supported EPS encryption algorithms (EEA identities, 0..=3).
    #[serde(default = "default_eea")]
    pub eea: Vec<u8>,
    /// Supported EPS integrity algorithms (EIA identities, 0..=3).
    #[serde(default = "default_eia")]
    pub eia: Vec<u8>,
    /// APN to request; absent asks for the subscription's default APN.
    #[serde(default)]
    pub apn: Option<String>,
    /// PDN type to request.
    #[serde(default)]
    pub pdn_type: PdnType,
}

fn default_eea() -> Vec<u8> {
    vec![0, 1, 2, 3]
}

fn default_eia() -> Vec<u8> {
    vec![0, 1, 2, 3]
}

impl LteUeConfig {
    /// The EPS NAS configuration, when a subscriber is configured.
    pub fn eps_nas(&self) -> Option<EpsNasConfig> {
        let s = self.subscriber.as_ref()?;
        let opc = match s.op_type {
            OpType::Opc => s.op,
            OpType::Op => nextgsim_crypto::milenage::compute_opc(&s.key, &s.op),
        };
        Some(EpsNasConfig {
            imsi: s.imsi.clone(),
            plmn: self.plmn,
            credentials: UsimCredentials { k: s.key, opc },
            initial_sqn: s.sqn,
            imeisv: s.imei_sv.clone(),
            eea: s.eea.clone(),
            eia: s.eia.clone(),
            apn: s.apn.clone(),
            pdn_type: s.pdn_type,
        })
    }
}

/// Configuration errors.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// Path that failed.
        path: String,
        /// Underlying error.
        source: std::io::Error,
    },
    /// The YAML did not deserialize.
    #[error("invalid YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// A value failed validation.
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

impl LteUeConfig {
    /// The search list as socket addresses.
    pub fn search_space(&self) -> Result<Vec<SocketAddr>, ConfigError> {
        self.enb_search_list
            .iter()
            .map(|s| {
                s.parse::<SocketAddr>()
                    .or_else(|_| {
                        s.parse::<IpAddr>()
                            .map(|ip| SocketAddr::new(ip, DEFAULT_RLS_PORT))
                    })
                    .map_err(|_| ConfigError::Invalid(format!("enb_search_list entry {s:?}")))
            })
            .collect()
    }
}

/// Load and validate a configuration file.
pub fn load_and_validate(path: impl AsRef<Path>) -> Result<LteUeConfig, ConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let c: LteUeConfig = serde_yaml::from_str(&text)?;
    validate(&c)?;
    Ok(c)
}

/// Load a configuration file, apply the `NEXTGSIM_*` environment overrides (see
/// [`apply_env_overrides`]), then validate the result.
pub fn load_with_env_and_validate(path: impl AsRef<Path>) -> Result<LteUeConfig, ConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut c: LteUeConfig = serde_yaml::from_str(&text)?;
    apply_env_overrides(&mut c, |k| std::env::var(k).ok())?;
    validate(&c)?;
    Ok(c)
}

/// Override deployment-dependent LTE UE fields from environment variables, so
/// one baked image config serves any test network (`Dockerfile.lte-ue-local`).
///
/// | Variable | Field(s) |
/// |---|---|
/// | `NEXTGSIM_UE_ENB_ADDR` | `enb_search_list` -- comma-separated `ip` or `ip:port` |
/// | `NEXTGSIM_PLMN` | `plmn`, written `MCC-MNC` |
/// | `NEXTGSIM_UE_IMSI` | `subscriber.imsi` |
/// | `NEXTGSIM_UE_KEY` | `subscriber.key`, 32 hex digits |
/// | `NEXTGSIM_UE_OPC` | `subscriber.op` with `op_type: Opc`, 32 hex digits |
/// | `NEXTGSIM_UE_APN` | `subscriber.apn` |
/// | `NEXTGSIM_UE_TUN_NAME` | `tun.name` (creates the `tun` section if absent) |
///
/// The subscriber variables need a `subscriber` section in the file to amend:
/// the remaining USIM fields (SQN, EEA/EIA, PDN type) have no sensible
/// defaults to invent, so an override without one is refused. `lookup` is
/// injected so tests need not mutate the process environment.
pub fn apply_env_overrides(
    c: &mut LteUeConfig,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<(), ConfigError> {
    use nextgsim_common::env_override::{parse_endpoints, parse_hex16, parse_plmn};
    let bad = |var: &str, e: String| ConfigError::Invalid(format!("{var}: {e}"));

    if let Some(v) = lookup("NEXTGSIM_UE_ENB_ADDR") {
        c.enb_search_list = parse_endpoints(&v, DEFAULT_RLS_PORT)
            .map_err(|e| bad("NEXTGSIM_UE_ENB_ADDR", e))?
            .iter()
            .map(ToString::to_string)
            .collect();
    }
    if let Some(v) = lookup("NEXTGSIM_PLMN") {
        c.plmn = parse_plmn(v.trim()).map_err(|e| bad("NEXTGSIM_PLMN", e))?;
    }
    let sub_vars = [
        "NEXTGSIM_UE_IMSI",
        "NEXTGSIM_UE_KEY",
        "NEXTGSIM_UE_OPC",
        "NEXTGSIM_UE_APN",
    ];
    if let Some(var) = sub_vars.iter().find(|v| lookup(v).is_some()) {
        let s = c.subscriber.as_mut().ok_or_else(|| {
            bad(
                var,
                "the configuration file has no `subscriber` section to amend".into(),
            )
        })?;
        if let Some(v) = lookup("NEXTGSIM_UE_IMSI") {
            s.imsi = v.trim().to_string();
        }
        if let Some(v) = lookup("NEXTGSIM_UE_KEY") {
            s.key = parse_hex16(v.trim()).map_err(|e| bad("NEXTGSIM_UE_KEY", e))?;
        }
        if let Some(v) = lookup("NEXTGSIM_UE_OPC") {
            s.op = parse_hex16(v.trim()).map_err(|e| bad("NEXTGSIM_UE_OPC", e))?;
            s.op_type = OpType::Opc;
        }
        if let Some(v) = lookup("NEXTGSIM_UE_APN") {
            s.apn = Some(v.trim().to_string());
        }
    }
    if let Some(v) = lookup("NEXTGSIM_UE_TUN_NAME") {
        c.tun.get_or_insert_with(TunSettings::default).name = Some(v.trim().to_string());
    }
    Ok(())
}

/// Validate a configuration.
pub fn validate(c: &LteUeConfig) -> Result<(), ConfigError> {
    let p = &c.plmn;
    if p.mcc > 999 || p.mnc > 999 || (!p.long_mnc && p.mnc > 99) {
        return Err(ConfigError::Invalid(format!("plmn {}-{}", p.mcc, p.mnc)));
    }
    if c.search_space()?.is_empty() {
        return Err(ConfigError::Invalid("enb_search_list is empty".into()));
    }
    if let Some(s) = &c.subscriber {
        let len = s.imsi.len();
        if !(6..=15).contains(&len) || !s.imsi.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ConfigError::Invalid(format!(
                "subscriber.imsi {:?}",
                s.imsi
            )));
        }
        if let Some(sv) = &s.imei_sv {
            if sv.len() != 16 || !sv.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ConfigError::Invalid(format!("subscriber.imei_sv {sv:?}")));
            }
        }
        if s.eea.iter().chain(&s.eia).any(|a| *a > 3) {
            return Err(ConfigError::Invalid(
                "subscriber.eea/eia: identities are 0..=3".into(),
            ));
        }
        if !s.eia.iter().any(|a| *a != 0) {
            // TS 33.401 §5.1.4.2: without a real EIA the UE could only attach
            // for an unauthenticated emergency call.
            return Err(ConfigError::Invalid(
                "subscriber.eia must include an algorithm other than EIA0".into(),
            ));
        }
        if s.sqn >= 1 << 48 {
            return Err(ConfigError::Invalid(
                "subscriber.sqn exceeds 48 bits".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| map.get(k).cloned()
    }

    fn with_subscriber() -> LteUeConfig {
        serde_yaml::from_str(
            "plmn: { mcc: 999, mnc: 70, long_mnc: false }\nenb_search_list: [\"127.0.0.1\"]\n\
             subscriber:\n  imsi: \"999700000000001\"\n  key: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]\n  \
             op: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]\n  op_type: Op\n  sqn: 0\n  eea: [0]\n  eia: [1]\n  pdn_type: ipv4\n",
        )
        .unwrap()
    }

    #[test]
    fn env_overrides_set_the_deployment_fields() {
        let mut c = with_subscriber();
        apply_env_overrides(
            &mut c,
            env(&[
                ("NEXTGSIM_UE_ENB_ADDR", "172.24.0.50"),
                ("NEXTGSIM_PLMN", "001-01"),
                ("NEXTGSIM_UE_IMSI", "001010000000002"),
                ("NEXTGSIM_UE_KEY", "465B5CE8B199B49FAA5F0A2EE238A6BC"),
                ("NEXTGSIM_UE_OPC", "E8ED289DEBA952E4283B54E88E6D834D"),
                ("NEXTGSIM_UE_APN", "internet"),
                ("NEXTGSIM_UE_TUN_NAME", "lteuetun0"),
            ]),
        )
        .unwrap();
        validate(&c).unwrap();
        assert_eq!(
            c.search_space().unwrap(),
            vec!["172.24.0.50:4997".parse().unwrap()]
        );
        assert_eq!(c.plmn, Plmn::new(1, 1, false));
        let s = c.subscriber.as_ref().unwrap();
        assert_eq!(s.imsi, "001010000000002");
        assert_eq!(s.key[0], 0x46);
        assert_eq!(s.op[15], 0x4D);
        assert_eq!(s.op_type, OpType::Opc);
        assert_eq!(s.apn.as_deref(), Some("internet"));
        assert_eq!(c.tun.as_ref().unwrap().name.as_deref(), Some("lteuetun0"));
    }

    #[test]
    fn no_env_leaves_the_file_untouched() {
        let mut c = with_subscriber();
        let before = format!("{c:?}");
        apply_env_overrides(&mut c, |_| None).unwrap();
        assert_eq!(format!("{c:?}"), before);
    }

    #[test]
    fn subscriber_override_needs_a_subscriber_section() {
        let mut c = with_subscriber();
        c.subscriber = None;
        let err = apply_env_overrides(&mut c, env(&[("NEXTGSIM_UE_KEY", "00")])).unwrap_err();
        assert!(err.to_string().contains("NEXTGSIM_UE_KEY"), "{err}");
    }

    #[test]
    fn malformed_env_is_refused_naming_the_variable() {
        for (var, val) in [
            ("NEXTGSIM_UE_ENB_ADDR", "enb.local"),
            ("NEXTGSIM_PLMN", "99970"),
            ("NEXTGSIM_UE_KEY", "465B"),
            ("NEXTGSIM_UE_OPC", "zz"),
        ] {
            let err = apply_env_overrides(&mut with_subscriber(), env(&[(var, val)])).unwrap_err();
            assert!(err.to_string().contains(var), "{var}: {err}");
        }
        // A bad IMSI is caught by the same validation a bad file gets.
        let mut c = with_subscriber();
        apply_env_overrides(&mut c, env(&[("NEXTGSIM_UE_IMSI", "12ab")])).unwrap();
        assert!(validate(&c).is_err());
    }

    #[test]
    fn search_list_accepts_bare_ips_and_ports() {
        let c: LteUeConfig = serde_yaml::from_str(
            "plmn: { mcc: 999, mnc: 70, long_mnc: false }\nenb_search_list: [\"127.0.0.1\", \"10.0.0.1:5000\"]\n",
        )
        .unwrap();
        validate(&c).unwrap();
        assert_eq!(
            c.search_space().unwrap(),
            vec![
                "127.0.0.1:4997".parse().unwrap(),
                "10.0.0.1:5000".parse().unwrap()
            ]
        );
    }

    #[test]
    fn empty_search_list_is_refused() {
        let c = LteUeConfig {
            plmn: Plmn::new(999, 70, false),
            enb_search_list: vec![],
            sti: None,
            position: [0; 3],
            subscriber: None,
            tun: None,
        };
        assert!(validate(&c).is_err());
    }

    #[test]
    fn the_shipped_config_loads_with_its_subscriber() {
        let c = load_and_validate(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../config/lte-ue.yaml"
        ))
        .unwrap();
        assert_eq!(c.eps_nas().unwrap().imsi, "999700000000001");
    }

    #[test]
    fn subscriber_yaml_follows_the_nr_ue_and_validates() {
        let c: LteUeConfig = serde_yaml::from_str(
            r#"
plmn: { mcc: 999, mnc: 70, long_mnc: false }
enb_search_list: ["127.0.0.1"]
subscriber:
  imsi: "999700000000001"
  key: [0x46, 0x5B, 0x5C, 0xE8, 0xB1, 0x99, 0xB4, 0x9F, 0xAA, 0x5F, 0x0A, 0x2E, 0xE2, 0x38, 0xA6, 0xBC]
  op: [0xE8, 0xED, 0x28, 0x9D, 0xEB, 0xA9, 0x52, 0xE4, 0x28, 0x3B, 0x54, 0xE8, 0x8E, 0x6D, 0x83, 0x4D]
  op_type: Opc
  imei_sv: "3534900698733019"
"#,
        )
        .unwrap();
        validate(&c).unwrap();
        let n = c.eps_nas().unwrap();
        assert_eq!(n.imsi, "999700000000001");
        assert_eq!(n.eia, vec![0, 1, 2, 3]);
        assert_eq!(n.pdn_type, PdnType::Ipv4);
        let mut bad = c.clone();
        bad.subscriber.as_mut().unwrap().eia = vec![0];
        assert!(validate(&bad).is_err());
        let mut bad = c;
        bad.subscriber.as_mut().unwrap().imsi = "99970x".into();
        assert!(validate(&bad).is_err());
    }
}
