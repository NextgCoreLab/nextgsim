//! Parsers for environment-variable overrides of node configuration.
//!
//! The container images (`Dockerfile.enb-local`, `Dockerfile.lte-ue-local`)
//! bake one YAML file and let a compose file adjust the handful of values that
//! depend on the deployment -- peer addresses, PLMN, subscriber credentials --
//! through `NEXTGSIM_*` variables, so a test network does not need its own copy
//! of every configuration file. Each binary owns the mapping from variable to
//! field; this module only holds the value parsers they share, so a malformed
//! value is reported the same way by every node.

use std::net::{IpAddr, SocketAddr};

use crate::Plmn;

/// Parse a PLMN written `MCC-MNC`, e.g. `999-70` or `001-001`.
///
/// A three-digit MNC (leading zeros included, as in `001-001`) sets
/// `long_mnc`, matching how the MNC is written on the air (TS 23.003 §2.2).
pub fn parse_plmn(value: &str) -> Result<Plmn, String> {
    let (mcc, mnc) = value
        .split_once('-')
        .ok_or_else(|| format!("{value:?} is not MCC-MNC"))?;
    let digits =
        |s: &str, lens: &[usize]| lens.contains(&s.len()) && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(mcc, &[3]) || !digits(mnc, &[2, 3]) {
        return Err(format!(
            "{value:?} is not MCC-MNC (3-digit MCC, 2- or 3-digit MNC)"
        ));
    }
    Ok(Plmn {
        mcc: mcc.parse().map_err(|_| format!("{value:?}: MCC"))?,
        mnc: mnc.parse().map_err(|_| format!("{value:?}: MNC"))?,
        long_mnc: mnc.len() == 3,
    })
}

/// Parse a 128-bit key or OPc written as 32 hex digits (an optional `0x` is
/// accepted).
pub fn parse_hex16(value: &str) -> Result<[u8; 16], String> {
    let hex = value.strip_prefix("0x").unwrap_or(value);
    if hex.len() != 32 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected 32 hex digits".to_string());
    }
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|_| "expected 32 hex digits".to_string())?;
    }
    Ok(out)
}

/// Parse an unsigned integer written in decimal or as `0x`-prefixed hex.
pub fn parse_u32(value: &str) -> Result<u32, String> {
    match value.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|_| format!("{value:?} is not a decimal or 0x-hex integer"))
}

/// Parse a comma-separated list of endpoints, each `ip` or `ip:port`; a bare
/// IP gets `default_port`.
pub fn parse_endpoints(value: &str, default_port: u16) -> Result<Vec<SocketAddr>, String> {
    let list: Vec<SocketAddr> = value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<SocketAddr>()
                .or_else(|_| {
                    s.parse::<IpAddr>()
                        .map(|ip| SocketAddr::new(ip, default_port))
                })
                .map_err(|_| format!("{s:?} is not an IP or IP:port"))
        })
        .collect::<Result<_, _>>()?;
    if list.is_empty() {
        return Err("empty list".to_string());
    }
    Ok(list)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plmn_two_and_three_digit_mnc() {
        assert_eq!(
            parse_plmn("999-70").unwrap(),
            Plmn {
                mcc: 999,
                mnc: 70,
                long_mnc: false
            }
        );
        assert_eq!(
            parse_plmn("001-001").unwrap(),
            Plmn {
                mcc: 1,
                mnc: 1,
                long_mnc: true
            }
        );
        for bad in ["99970", "99-70", "999-7", "999-7000", "abc-70", ""] {
            assert!(parse_plmn(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn hex16_round_trip() {
        let k = parse_hex16("465B5CE8B199B49FAA5F0A2EE238A6BC").unwrap();
        assert_eq!(k[0], 0x46);
        assert_eq!(k[15], 0xBC);
        assert_eq!(
            parse_hex16("0x465b5ce8b199b49faa5f0a2ee238a6bc").unwrap(),
            k
        );
        assert!(parse_hex16("465B").is_err());
        assert!(parse_hex16("Z65B5CE8B199B49FAA5F0A2EE238A6BC").is_err());
    }

    #[test]
    fn u32_dec_and_hex() {
        assert_eq!(parse_u32("411").unwrap(), 411);
        assert_eq!(parse_u32("0x19B").unwrap(), 0x19B);
        assert!(parse_u32("0xZZ").is_err());
    }

    #[test]
    fn endpoints_default_port() {
        let v = parse_endpoints("172.24.0.5, 172.24.0.15:36413", 36412).unwrap();
        assert_eq!(v[0], "172.24.0.5:36412".parse().unwrap());
        assert_eq!(v[1], "172.24.0.15:36413".parse().unwrap());
        assert!(parse_endpoints(" , ", 1).is_err());
        assert!(parse_endpoints("not-an-ip", 1).is_err());
    }
}
