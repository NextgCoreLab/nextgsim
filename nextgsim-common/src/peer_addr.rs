//! Peer endpoints that may be named by hostname (issue #286).
//!
//! The gNB's AMF address and the UE's `gnb_search_list` used to be literal IPs
//! only. On Kubernetes a peer pod comes back at a NEW address after a restart,
//! so a literal resolved once (an init container) strands the dependent for
//! good. A [`PeerAddr`] keeps the name, and the caller re-resolves it whenever
//! it reconnects.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

/// How long one resolution may take before it counts as a failure.
///
/// Bounded because the callers resolve from inside their task loop: an
/// unresponsive resolver must not stall heartbeats or NGAP for the system
/// default (which can be tens of seconds).
pub const RESOLVE_TIMEOUT: Duration = Duration::from_secs(2);

/// A configured peer: a literal endpoint, or a hostname plus port.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PeerAddr {
    /// `10.0.0.1`, `10.0.0.1:4997`, `[::1]:38412`
    Literal(SocketAddr),
    /// `amf.nextg-system.svc.cluster.local`, `gnb:4997`
    Host(String, u16),
}

impl PeerAddr {
    /// Parse an entry: a bare IP or hostname takes `default_port`; `IP:port`,
    /// `[v6]:port` and `host:port` carry their own.
    ///
    /// Nothing is resolved here, so a name whose DNS record does not exist yet
    /// (a peer pod that is not Ready) parses fine and is resolved later.
    pub fn parse(entry: &str, default_port: u16) -> Result<Self, String> {
        let entry = entry.trim();
        if entry.is_empty() {
            return Err("entry is empty".to_string());
        }
        if let Ok(ip) = entry.parse::<IpAddr>() {
            return Ok(Self::Literal(SocketAddr::new(ip, default_port)));
        }
        if let Ok(sa) = entry.parse::<SocketAddr>() {
            return Ok(Self::Literal(sa));
        }
        let (host, port) = match entry.rsplit_once(':') {
            Some((h, p)) => (
                h,
                p.parse::<u16>()
                    .map_err(|_| format!("{entry:?} has an invalid port"))?,
            ),
            None => (entry, default_port),
        };
        if !is_valid_hostname(host) {
            return Err(format!(
                "{entry:?} is neither an IP address, IP:port, nor a valid hostname[:port]"
            ));
        }
        Ok(Self::Host(host.to_string(), port))
    }

    /// The port this peer is reached on.
    pub fn port(&self) -> u16 {
        match self {
            Self::Literal(sa) => sa.port(),
            Self::Host(_, port) => *port,
        }
    }

    /// Whether this is a hostname, i.e. whether resolving it again can give a
    /// different answer.
    pub fn is_hostname(&self) -> bool {
        matches!(self, Self::Host(..))
    }

    /// Resolve to the peer's current addresses, IPv4 first.
    ///
    /// A literal resolves to itself without touching DNS. A hostname is looked
    /// up afresh on every call, bounded by [`RESOLVE_TIMEOUT`].
    pub async fn resolve(&self) -> std::io::Result<Vec<SocketAddr>> {
        match self {
            Self::Literal(sa) => Ok(vec![*sa]),
            Self::Host(host, port) => {
                let lookup = tokio::net::lookup_host((host.as_str(), *port));
                let addrs = tokio::time::timeout(RESOLVE_TIMEOUT, lookup)
                    .await
                    .map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!("resolving {host} timed out"),
                        )
                    })??;
                let mut addrs: Vec<SocketAddr> = addrs.collect();
                addrs.sort_by_key(|a| !a.is_ipv4());
                addrs.dedup();
                if addrs.is_empty() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("{host} resolved to no address"),
                    ));
                }
                Ok(addrs)
            }
        }
    }
}

impl std::fmt::Display for PeerAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Literal(sa) => write!(f, "{sa}"),
            Self::Host(host, port) => write!(f, "{host}:{port}"),
        }
    }
}

/// RFC 1123 hostname shape: dot-separated labels of letters, digits and
/// hyphens (plus `_`, which Kubernetes SRV-style names use), each 1..=63
/// characters, not starting or ending with a hyphen.
fn is_valid_hostname(host: &str) -> bool {
    !host.is_empty()
        && host.len() <= 253
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_and_hostnames_parse() {
        assert_eq!(
            PeerAddr::parse("10.0.0.1", 4997),
            Ok(PeerAddr::Literal("10.0.0.1:4997".parse().unwrap()))
        );
        assert_eq!(
            PeerAddr::parse("10.0.0.1:5000", 4997),
            Ok(PeerAddr::Literal("10.0.0.1:5000".parse().unwrap()))
        );
        assert_eq!(
            PeerAddr::parse("::1", 38412),
            Ok(PeerAddr::Literal("[::1]:38412".parse().unwrap()))
        );
        assert_eq!(
            PeerAddr::parse("[::1]:9", 38412),
            Ok(PeerAddr::Literal("[::1]:9".parse().unwrap()))
        );
        assert_eq!(
            PeerAddr::parse("amf.nextg-system.svc.cluster.local", 38412),
            Ok(PeerAddr::Host(
                "amf.nextg-system.svc.cluster.local".into(),
                38412
            ))
        );
        assert_eq!(
            PeerAddr::parse(" gnb:4998 ", 4997),
            Ok(PeerAddr::Host("gnb".into(), 4998))
        );
    }

    #[test]
    fn garbage_is_rejected_with_the_entry_named() {
        for bad in ["", "gnb:port", "bad host", "-gnb", "a..b", "gnb:70000"] {
            let err = PeerAddr::parse(bad, 4997).expect_err(bad);
            assert!(!err.is_empty());
        }
        assert!(PeerAddr::parse("bad host", 1)
            .unwrap_err()
            .contains("bad host"));
    }

    #[tokio::test]
    async fn a_literal_resolves_to_itself_and_a_hostname_through_dns() {
        let lit = PeerAddr::parse("192.0.2.7:9", 1).unwrap();
        assert_eq!(
            lit.resolve().await.unwrap(),
            vec!["192.0.2.7:9".parse().unwrap()]
        );

        let host = PeerAddr::parse("localhost:4321", 1).unwrap();
        let addrs = host.resolve().await.expect("localhost resolves");
        assert!(addrs
            .iter()
            .all(|a| a.port() == 4321 && a.ip().is_loopback()));

        let missing = PeerAddr::parse("no-such-peer.invalid", 1).unwrap();
        assert!(
            missing.resolve().await.is_err(),
            "an unknown name is an error, not a panic"
        );
    }
}
