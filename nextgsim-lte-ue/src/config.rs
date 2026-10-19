//! LTE UE configuration. Shape follows the NR UE's (`plmn`, a search list of
//! RLS addresses), so the two can sit side by side in one deployment.

use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;

use nextgsim_common::Plmn;
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

/// Validate a configuration.
pub fn validate(c: &LteUeConfig) -> Result<(), ConfigError> {
    let p = &c.plmn;
    if p.mcc > 999 || p.mnc > 999 || (!p.long_mnc && p.mnc > 99) {
        return Err(ConfigError::Invalid(format!("plmn {}-{}", p.mcc, p.mnc)));
    }
    if c.search_space()?.is_empty() {
        return Err(ConfigError::Invalid("enb_search_list is empty".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };
        assert!(validate(&c).is_err());
    }
}
