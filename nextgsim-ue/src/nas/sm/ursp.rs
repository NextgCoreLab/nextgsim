//! URSP rule evaluation — TS 24.526 §5.2, TS 23.503 §6.6.2.3.
//!
//! The UE has always decoded and stored the URSP rules a PCF delivers, and has
//! always accepted `ursp_rules` in its own configuration, but nothing ever
//! *evaluated* them: PDU sessions were established purely from the static
//! [`SmSessionParams`], so an operator's slice/DNN steering policy had no effect
//! on which session an application's traffic bound to (#47).
//!
//! This module is the missing engine. Given an [`ApplicationDescriptor`] it
//! walks the rules in ascending precedence, matches each traffic descriptor, and
//! returns the route selection descriptor of the first match — falling back to
//! the default (match-all) rule, per §5.2.
//!
//! # Off by default, and switched at RUNTIME
//!
//! Evaluation changes which DNN and S-NSSAI a session is established with, so it
//! is gated on `UeConfig::ursp_evaluation`, default `false`. **A runtime switch,
//! not a cargo feature**, against #47's own suggested approach: CI runs
//! `cargo test --workspace` with default features, so a feature-gated engine
//! would ship without ever being compiled by the gate meant to cover it. Both
//! states of the switch are inside the test suite.
//!
//! # One engine, two rule sources
//!
//! Network-delivered rules arrive as the wire type
//! [`nextgsim_nas::messages::mm::ue_policy::UrspRule`]. Configured rules use the
//! simpler YAML shape [`nextgsim_common::config::UrspRule`]. Rather than two
//! matchers, configured rules are **converted into the wire type** so exactly one
//! evaluation path exists — a second matcher is a second set of precedence bugs.

use tracing::{debug, info};

use nextgsim_common::config::{
    PduSessionType as ConfigPduSessionType, UeConfig, UrspRule as ConfigUrspRule,
};
use nextgsim_nas::ies::ie1::{PduSessionType, SscMode};
use nextgsim_nas::messages::mm::ue_policy::{
    RouteSelectionDescriptor, RouteSelectionDescriptorComponent, TrafficDescriptorComponent,
    UrspRule,
};
use nextgsim_nas::messages::sm::{PduSessionTypeValue, SscModeValue};

use super::SmSessionParams;

/// The "application information" §5.2 matches a traffic descriptor against.
///
/// Every field is optional because a real detection sees only some of them: a
/// TUN write yields addresses and ports but no OS App Id, while an
/// application-level hook yields the App Id and no packet at all. A component
/// whose corresponding field is `None` does **not** match — see
/// [`Self::matches_component`] for why that direction is the safe one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApplicationDescriptor {
    /// OS Id (RFC 4122 UUID octets) of the running OS, if known.
    pub os_id: Option<[u8; 16]>,
    /// OS App Id of the detected application, if known.
    pub os_app_id: Option<Vec<u8>>,
    /// DNN the application asked for, if it named one.
    pub dnn: Option<String>,
    /// Destination FQDN the application resolved, if known.
    pub fqdn: Option<String>,
    /// Remote IPv4 address of the flow, if known.
    pub ipv4: Option<[u8; 4]>,
    /// Remote IPv6 address of the flow, if known.
    pub ipv6: Option<[u8; 16]>,
    /// IP protocol / next-header value, if known.
    pub protocol: Option<u8>,
    /// Remote port, if known.
    pub port: Option<u16>,
}

impl ApplicationDescriptor {
    /// Classify an uplink IP packet into the application information URSP matches
    /// against (issue #97).
    ///
    /// This is the traffic-classification hook the URSP engine was missing. Before
    /// it, the only descriptor production could build named a DNN and nothing else,
    /// so five of the eight traffic-descriptor component types -- IPv4 and IPv6
    /// remote address, protocol identifier, single remote port and port range --
    /// could never match however the network configured them.
    ///
    /// Reads the **remote** (destination) address and port, which is what a
    /// traffic descriptor names: TS 24.526 §5.2's descriptors are written from the
    /// UE's point of view, so "remote" is the far end of an uplink packet.
    ///
    /// `None` when the bytes are not a packet this can classify: too short, an
    /// unrecognised IP version, or a truncated transport header. `None` rather than
    /// a partially-filled descriptor, because a descriptor with a plausible address
    /// and a garbage port would match a rule the flow does not belong to.
    ///
    /// ## What it cannot fill, and why
    ///
    /// - **`fqdn`**: this UE performs no DNS interception, so a resolved name is
    ///   not observable from a packet. `DestinationFqdn` components remain
    ///   decode-and-match-only.
    /// - **`os_id` / `os_app_id`**: not derivable from a packet at all. They come
    ///   from configuration instead (`ursp_os_id`, `ursp_os_app_id`), which is the
    ///   "pretend this app is running" hook -- an honest stand-in for an OS the
    ///   simulator does not have, rather than a value invented per packet.
    pub fn from_uplink_packet(packet: &[u8]) -> Option<Self> {
        let version = packet.first()? >> 4;
        let (protocol, remote_ipv4, remote_ipv6, transport) = match version {
            4 => {
                // IPv4 (RFC 791): IHL is the low nibble of octet 0, in 32-bit words.
                let ihl = usize::from(packet[0] & 0x0F) * 4;
                if ihl < 20 || packet.len() < ihl {
                    return None;
                }
                let protocol = *packet.get(9)?;
                let dst: [u8; 4] = packet.get(16..20)?.try_into().ok()?;
                (protocol, Some(dst), None, &packet[ihl..])
            }
            6 => {
                // IPv6 (RFC 8200): a fixed 40-octet header. Extension headers are
                // NOT walked -- a packet carrying one is left unclassified rather
                // than having its Next Header read as a transport protocol, which
                // would report a routing header as if it were TCP.
                if packet.len() < 40 {
                    return None;
                }
                let protocol = packet[6];
                let dst: [u8; 16] = packet.get(24..40)?.try_into().ok()?;
                (protocol, None, Some(dst), &packet[40..])
            }
            _ => return None,
        };

        // The destination port, for the transports that have one. Anything else
        // classifies with a protocol and no port, which still matches a
        // protocol-keyed rule.
        const TCP: u8 = 6;
        const UDP: u8 = 17;
        let port = match protocol {
            TCP | UDP if transport.len() >= 4 => {
                Some(u16::from_be_bytes([transport[2], transport[3]]))
            }
            TCP | UDP => return None, // truncated transport header: not classifiable
            _ => None,
        };

        Some(Self {
            os_id: None,
            os_app_id: None,
            dnn: None,
            fqdn: None,
            ipv4: remote_ipv4,
            ipv6: remote_ipv6,
            protocol: Some(protocol),
            port,
        })
    }

    /// The OS identity the configuration declares, decoded (issue #97).
    ///
    /// Returns `(os_id, os_app_id)`. A `ursp_os_id` that is not 32 hex characters
    /// yields `None` for the id with a warning rather than a partial UUID: half a
    /// UUID identifies a different OS.
    pub fn configured_os_identity(
        config: &nextgsim_common::config::UeConfig,
    ) -> (Option<[u8; 16]>, Option<Vec<u8>>) {
        let os_app_id = config
            .ursp_os_app_id
            .as_ref()
            .map(|id| id.as_bytes().to_vec());

        let os_id = config.ursp_os_id.as_ref().and_then(|hex| {
            let hex = hex.trim().replace('-', "");
            if hex.len() != 32 {
                tracing::warn!(
                    "ursp_os_id must be 32 hex characters (a UUID); {} given -- OS Id not reported",
                    hex.len()
                );
                return None;
            }
            let mut octets = [0u8; 16];
            for (index, octet) in octets.iter_mut().enumerate() {
                *octet = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).ok()?;
            }
            Some(octets)
        });

        (os_id, os_app_id)
    }

    /// Add the configured OS identity to a descriptor (issue #97).
    ///
    /// The simulator has no OS and detects no applications, so an `OsIdOsAppId`
    /// component can only ever match something the operator declared. Applying it
    /// here rather than inside [`Self::from_uplink_packet`] keeps the classifier a
    /// pure function of the packet.
    pub fn with_os_identity(mut self, os_id: Option<[u8; 16]>, os_app_id: Option<Vec<u8>>) -> Self {
        self.os_id = os_id;
        self.os_app_id = os_app_id;
        self
    }

    /// A descriptor naming only a DNN — what the configured default sessions
    /// have to offer, and enough to exercise DNN-keyed steering rules.
    pub fn for_dnn(dnn: impl Into<String>) -> Self {
        Self {
            dnn: Some(dnn.into()),
            ..Default::default()
        }
    }

    /// A descriptor naming an application by its OS App Id.
    pub fn for_app_id(os_app_id: impl Into<Vec<u8>>) -> Self {
        Self {
            os_app_id: Some(os_app_id.into()),
            ..Default::default()
        }
    }

    /// Whether one traffic-descriptor component matches this application.
    ///
    /// An `None` field never matches a component that tests it. That direction
    /// matters: the opposite convention ("unknown matches anything") would make
    /// a descriptor with no information at all match every rule, so the
    /// highest-priority rule in the URSP would capture all traffic regardless of
    /// what it was written for.
    fn matches_component(&self, component: &TrafficDescriptorComponent) -> bool {
        match component {
            // §4.2.2.2: the match-all descriptor identifies the default rule.
            TrafficDescriptorComponent::MatchAll => true,
            TrafficDescriptorComponent::OsIdOsAppId { os_id, os_app_id } => {
                // The OS Id is compared only when the application reported one:
                // a UE that does not model an OS identity would otherwise never
                // match a rule the PCF wrote for its app.
                let os_ok = self.os_id.map(|own| own == *os_id).unwrap_or(true);
                os_ok && self.os_app_id.as_deref() == Some(os_app_id.as_slice())
            }
            TrafficDescriptorComponent::Dnn(dnn) => self.dnn.as_deref() == Some(dnn.as_str()),
            TrafficDescriptorComponent::DestinationFqdn(fqdn) => {
                // FQDN comparison is case-insensitive (DNS names are).
                self.fqdn
                    .as_deref()
                    .is_some_and(|own| own.eq_ignore_ascii_case(fqdn))
            }
            TrafficDescriptorComponent::Ipv4RemoteAddress { addr, mask } => {
                self.ipv4.is_some_and(|own| {
                    own.iter()
                        .zip(addr.iter())
                        .zip(mask.iter())
                        .all(|((o, a), m)| (o & m) == (a & m))
                })
            }
            TrafficDescriptorComponent::Ipv6RemoteAddress { addr, prefix_len } => {
                self.ipv6.is_some_and(|own| {
                    let bits = usize::from((*prefix_len).min(128));
                    let full = bits / 8;
                    if own[..full] != addr[..full] {
                        return false;
                    }
                    let rem = bits % 8;
                    if rem == 0 {
                        return true;
                    }
                    let mask = 0xFFu8 << (8 - rem);
                    (own[full] & mask) == (addr[full] & mask)
                })
            }
            TrafficDescriptorComponent::ProtocolIdentifier(proto) => self.protocol == Some(*proto),
            TrafficDescriptorComponent::SingleRemotePort(port) => self.port == Some(*port),
            TrafficDescriptorComponent::RemotePortRange { low, high } => {
                self.port.is_some_and(|p| p >= *low && p <= *high)
            }
        }
    }

    /// Whether a rule's whole traffic descriptor matches.
    ///
    /// **All** components must match. TS 24.526 §5.2 describes a traffic
    /// descriptor as the set of conditions identifying the application, so the
    /// components are a conjunction; treating them as a disjunction would make a
    /// rule scoped to "this app on this port" fire for the app on any port.
    fn matches(&self, descriptor: &[TrafficDescriptorComponent]) -> bool {
        !descriptor.is_empty() && descriptor.iter().all(|c| self.matches_component(c))
    }
}

/// A rule's traffic descriptor is the match-all one, i.e. this is the default
/// rule (TS 24.526 §4.2.2.2).
fn is_default_rule(rule: &UrspRule) -> bool {
    rule.traffic_descriptor
        .iter()
        .any(|c| matches!(c, TrafficDescriptorComponent::MatchAll))
}

/// The outcome of an evaluation: which rule matched, and by which route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrspMatch {
    /// Precedence of the rule that matched (lower value = higher priority).
    pub precedence: u8,
    /// Whether the match came from the default (match-all) rule.
    pub is_default: bool,
    /// The selected route selection descriptor.
    pub route: RouteSelectionDescriptor,
}

impl UrspMatch {
    /// Apply the matched route to `base`, returning the session parameters to
    /// establish (or to look for when reusing a session).
    ///
    /// Components the route does not carry leave `base` untouched: a route that
    /// names only an S-NSSAI steers the slice and inherits the configured DNN,
    /// which is what §5.2's "association using the matched route selection
    /// descriptor" means — the RSD is an override set, not a whole config.
    pub fn apply_to(&self, base: &SmSessionParams) -> SmSessionParams {
        let mut params = base.clone();
        for component in &self.route.components {
            match component {
                RouteSelectionDescriptorComponent::Dnn(dnn) => {
                    params.dnn = Some(dnn.clone());
                }
                RouteSelectionDescriptorComponent::SNssai { sst, sd } => {
                    let mut value = vec![*sst];
                    if let Some(sd) = sd {
                        value.extend_from_slice(sd);
                    }
                    params.s_nssai = Some(value);
                }
                RouteSelectionDescriptorComponent::SscMode(mode) => {
                    params.ssc_mode = match mode {
                        SscMode::SscMode1 => SscModeValue::SscMode1,
                        SscMode::SscMode2 => SscModeValue::SscMode2,
                        SscMode::SscMode3 => SscModeValue::SscMode3,
                    };
                }
                RouteSelectionDescriptorComponent::PduSessionType(pdu_type) => {
                    params.session_type = match pdu_type {
                        PduSessionType::Ipv4 => PduSessionTypeValue::Ipv4,
                        PduSessionType::Ipv6 => PduSessionTypeValue::Ipv6,
                        PduSessionType::Ipv4v6 => PduSessionTypeValue::Ipv4v6,
                        PduSessionType::Unstructured => PduSessionTypeValue::Unstructured,
                        PduSessionType::Ethernet => PduSessionTypeValue::Ethernet,
                    };
                }
                RouteSelectionDescriptorComponent::PreferredAccessType(_) => {
                    // 3GPP vs non-3GPP access preference. This UE has one access
                    // and no ATSSS, so honouring it would mean either ignoring
                    // the rule or refusing a session the UE can serve. Recorded
                    // rather than acted on.
                    debug!("URSP route names a preferred access type; this UE has one access");
                }
            }
        }
        params
    }
}

/// The URSP rules this UE evaluates, and whether evaluation is on.
///
/// Holds configured and network-delivered rules in one list. Network-delivered
/// rules are installed by [`Self::set_delivered_rules`] each time the stored
/// policy changes, replacing the previous delivered set — a PCF's URSP is the
/// whole policy for that PLMN, not a delta.
#[derive(Debug, Clone, Default)]
pub struct UrspPolicy {
    enabled: bool,
    configured: Vec<UrspRule>,
    delivered: Vec<UrspRule>,
}

impl UrspPolicy {
    /// Build the policy from a UE configuration, converting its YAML rules into
    /// the wire rule type so there is one evaluation path.
    pub fn from_config(config: &UeConfig) -> Self {
        let configured: Vec<UrspRule> = config
            .ursp_rules
            .iter()
            .filter_map(convert_config_rule)
            .collect();
        if config.ursp_evaluation {
            info!(
                "URSP evaluation ENABLED with {} configured rule(s) (TS 24.526 §5.2)",
                configured.len()
            );
        } else if !configured.is_empty() {
            info!(
                "{} URSP rule(s) configured but ursp_evaluation is off: session parameters come \
                 from the static session config",
                configured.len()
            );
        }
        Self {
            enabled: config.ursp_evaluation,
            configured,
            delivered: Vec::new(),
        }
    }

    /// Whether evaluation is switched on.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Install the network-delivered rules, replacing any previous set.
    pub fn set_delivered_rules(&mut self, rules: Vec<UrspRule>) {
        if !rules.is_empty() || !self.delivered.is_empty() {
            info!(
                "URSP: {} network-delivered rule(s) installed (was {})",
                rules.len(),
                self.delivered.len()
            );
        }
        self.delivered = rules;
    }

    /// Total rule count across both sources (observability / test hook).
    pub fn rule_count(&self) -> usize {
        self.configured.len() + self.delivered.len()
    }

    /// Evaluate the rules for `app` per TS 24.526 §5.2.
    ///
    /// Returns `None` when evaluation is off, or when no rule matches and there
    /// is no default rule — in which case the caller keeps its static parameters,
    /// which is the pre-#47 behaviour and the only safe answer: §5.2 gives no
    /// route for an unmatched application without a default rule.
    pub fn evaluate(&self, app: &ApplicationDescriptor) -> Option<UrspMatch> {
        if !self.enabled {
            return None;
        }

        // Network-delivered rules take priority over configured ones at equal
        // precedence: the PCF's policy outranks a local default. The sort is
        // stable, so ordering the sources this way is what expresses that.
        let mut candidates: Vec<&UrspRule> = self
            .delivered
            .iter()
            .chain(self.configured.iter())
            .collect();
        // §5.2: "in increasing order of precedence values".
        candidates.sort_by_key(|r| r.precedence);

        // The default rule is excluded from the ordered walk and applied only as
        // a fallback, per §5.2 ("except the default URSP rule"). Without this a
        // default rule with a low precedence value would capture every
        // application before any specific rule was consulted.
        for rule in candidates.iter().filter(|r| !is_default_rule(r)) {
            if app.matches(&rule.traffic_descriptor) {
                if let Some(route) = select_route(rule) {
                    debug!(
                        "URSP match: rule precedence {} for {app:?}",
                        rule.precedence
                    );
                    return Some(UrspMatch {
                        precedence: rule.precedence,
                        is_default: false,
                        route,
                    });
                }
                // A matched rule whose routes are all unusable does NOT stop the
                // walk: §5.2 has the UE continue when it cannot use the selected
                // descriptor, and stopping would strand the application on the
                // static config while a lower-priority rule could have served it.
                debug!(
                    "URSP rule precedence {} matched but carries no usable route; continuing",
                    rule.precedence
                );
            }
        }

        let default = candidates.iter().find(|r| is_default_rule(r))?;
        let route = select_route(default)?;
        debug!("URSP: falling back to the default rule (§5.2)");
        Some(UrspMatch {
            precedence: default.precedence,
            is_default: true,
            route,
        })
    }

    /// Resolve the session parameters for `app`, or return `base` unchanged.
    pub fn resolve(&self, app: &ApplicationDescriptor, base: &SmSessionParams) -> SmSessionParams {
        match self.evaluate(app) {
            Some(matched) => {
                let params = matched.apply_to(base);
                if params != *base {
                    info!(
                        "URSP steered the session: DNN {:?} -> {:?}, S-NSSAI {:?} -> {:?} \
                         (rule precedence {}{})",
                        base.dnn,
                        params.dnn,
                        base.s_nssai,
                        params.s_nssai,
                        matched.precedence,
                        if matched.is_default { ", default" } else { "" }
                    );
                }
                params
            }
            None => base.clone(),
        }
    }
}

/// The route selection descriptor to use from a matched rule: the lowest
/// precedence value among those carrying at least one component.
///
/// TS 24.526 §5.2 walks the RSDs in increasing precedence order too, so the
/// lowest usable one is the selection.
fn select_route(rule: &UrspRule) -> Option<RouteSelectionDescriptor> {
    rule.route_selection_descriptors
        .iter()
        .filter(|rsd| !rsd.components.is_empty())
        .min_by_key(|rsd| rsd.precedence)
        .cloned()
}

/// Convert a configured YAML rule into the wire rule type.
///
/// The config's `traffic_descriptor` is a single string, which the wire form has
/// no direct equivalent for, so this is a **nextgsim configuration convenience**
/// with an explicit spelling rather than a spec form:
///
/// | `traffic_descriptor` | Wire component |
/// | --- | --- |
/// | `*` or `match-all` | `MatchAll` (makes this the default rule) |
/// | `dnn:<name>` | `Dnn(<name>)` |
/// | `fqdn:<name>` | `DestinationFqdn(<name>)` |
/// | `ipv4:<a.b.c.d>` or `ipv4:<a.b.c.d>/<len>` | `Ipv4RemoteAddress` |
/// | `ipv6:<addr>/<len>` | `Ipv6RemoteAddress` |
/// | `proto:<0-255>` | `ProtocolIdentifier` |
/// | `port:<n>` or `port:<lo>-<hi>` | `SingleRemotePort` / `RemotePortRange` |
/// | anything else | `OsIdOsAppId` with a zero OS Id and the string as App Id |
///
/// The four flow-keyed spellings arrived with issue #97's traffic classification.
/// Without them the newly-reachable matchers would serve only PCF-delivered rules,
/// so an operator could not configure the steering the classifier makes possible.
///
/// Returns `None` for a rule with no usable route selection descriptor: a rule
/// that can match but cannot steer is not a rule, and silently keeping it would
/// let it shadow a later rule that could have served the application.
/// Parse an `ipv4:` config descriptor: `a.b.c.d` or `a.b.c.d/len`.
///
/// A bare address means an exact match, i.e. a full 32-bit mask. Defaulting a
/// missing prefix to 0 instead would turn "this host" into "every host".
fn parse_ipv4_descriptor(spec: &str) -> Option<TrafficDescriptorComponent> {
    let (address, prefix) = match spec.trim().split_once('/') {
        Some((address, len)) => (address, len.trim().parse::<u8>().ok()?),
        None => (spec.trim(), 32),
    };
    if prefix > 32 {
        return None;
    }
    let octets: Vec<u8> = address
        .split('.')
        .map(|part| part.trim().parse::<u8>().ok())
        .collect::<Option<Vec<u8>>>()?;
    let addr: [u8; 4] = octets.try_into().ok()?;

    // A prefix length as a big-endian mask. Shifting a u32 by 32 is undefined, so
    // the zero-prefix case is handled explicitly rather than by the shift.
    let mask_bits: u32 = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix))
    };
    Some(TrafficDescriptorComponent::Ipv4RemoteAddress {
        addr,
        mask: mask_bits.to_be_bytes(),
    })
}

/// Parse an `ipv6:` config descriptor: `<addr>/<len>`, or a bare address meaning
/// an exact /128 match.
///
/// Accepts the full colon-hex form with at most one `::` run, which is what an
/// operator writes. A malformed address yields `None` and the rule is dropped.
fn parse_ipv6_descriptor(spec: &str) -> Option<TrafficDescriptorComponent> {
    let (address, prefix) = match spec.trim().split_once('/') {
        Some((address, len)) => (address, len.trim().parse::<u8>().ok()?),
        None => (spec.trim(), 128),
    };
    if prefix > 128 {
        return None;
    }
    let addr = parse_ipv6_address(address)?;
    Some(TrafficDescriptorComponent::Ipv6RemoteAddress {
        addr,
        prefix_len: prefix,
    })
}

/// Parse a colon-hex IPv6 address into its 16 octets.
fn parse_ipv6_address(text: &str) -> Option<[u8; 16]> {
    let text = text.trim();
    // At most one "::" run, per RFC 4291 §2.2.
    let (head, tail) = match text.split_once("::") {
        Some((head, tail)) => {
            if tail.contains("::") {
                return None;
            }
            (head, Some(tail))
        }
        None => (text, None),
    };

    let parse_groups = |part: &str| -> Option<Vec<u16>> {
        if part.is_empty() {
            return Some(Vec::new());
        }
        part.split(':')
            .map(|group| u16::from_str_radix(group, 16).ok())
            .collect()
    };

    let head_groups = parse_groups(head)?;
    let tail_groups = match tail {
        Some(tail) => parse_groups(tail)?,
        None => Vec::new(),
    };

    let mut groups = [0u16; 8];
    if tail.is_none() {
        if head_groups.len() != 8 {
            return None;
        }
        groups.copy_from_slice(&head_groups);
    } else {
        if head_groups.len() + tail_groups.len() > 8 {
            return None;
        }
        groups[..head_groups.len()].copy_from_slice(&head_groups);
        let start = 8 - tail_groups.len();
        groups[start..].copy_from_slice(&tail_groups);
    }

    let mut octets = [0u8; 16];
    for (index, group) in groups.iter().enumerate() {
        octets[index * 2..index * 2 + 2].copy_from_slice(&group.to_be_bytes());
    }
    Some(octets)
}

/// Parse a `port:` config descriptor: `<n>` or `<lo>-<hi>`.
///
/// A reversed range is rejected rather than swapped: `port:500-100` is a typo, and
/// silently reading it as 100-500 applies a policy the operator did not write.
fn parse_port_descriptor(spec: &str) -> Option<TrafficDescriptorComponent> {
    let spec = spec.trim();
    match spec.split_once('-') {
        Some((low, high)) => {
            let low = low.trim().parse::<u16>().ok()?;
            let high = high.trim().parse::<u16>().ok()?;
            if low > high {
                return None;
            }
            Some(TrafficDescriptorComponent::RemotePortRange { low, high })
        }
        None => Some(TrafficDescriptorComponent::SingleRemotePort(
            spec.parse::<u16>().ok()?,
        )),
    }
}

fn convert_config_rule(rule: &ConfigUrspRule) -> Option<UrspRule> {
    let descriptor = rule.traffic_descriptor.trim();
    let component = if descriptor == "*" || descriptor.eq_ignore_ascii_case("match-all") {
        TrafficDescriptorComponent::MatchAll
    } else if let Some(dnn) = descriptor.strip_prefix("dnn:") {
        TrafficDescriptorComponent::Dnn(dnn.to_string())
    } else if let Some(fqdn) = descriptor.strip_prefix("fqdn:") {
        TrafficDescriptorComponent::DestinationFqdn(fqdn.to_string())
    } else if let Some(spec) = descriptor.strip_prefix("ipv4:") {
        // A rule whose address does not parse is DROPPED rather than widened to
        // match-all: a malformed address in an allow-style rule that fell back to
        // matching everything would steer every flow.
        match parse_ipv4_descriptor(spec) {
            Some(component) => component,
            None => {
                debug!("URSP config: ignoring rule with unparsable ipv4 descriptor '{spec}'");
                return None;
            }
        }
    } else if let Some(spec) = descriptor.strip_prefix("ipv6:") {
        match parse_ipv6_descriptor(spec) {
            Some(component) => component,
            None => {
                debug!("URSP config: ignoring rule with unparsable ipv6 descriptor '{spec}'");
                return None;
            }
        }
    } else if let Some(spec) = descriptor.strip_prefix("proto:") {
        match spec.trim().parse::<u8>() {
            Ok(proto) => TrafficDescriptorComponent::ProtocolIdentifier(proto),
            Err(_) => {
                debug!("URSP config: ignoring rule with unparsable protocol '{spec}'");
                return None;
            }
        }
    } else if let Some(spec) = descriptor.strip_prefix("port:") {
        match parse_port_descriptor(spec) {
            Some(component) => component,
            None => {
                debug!("URSP config: ignoring rule with unparsable port descriptor '{spec}'");
                return None;
            }
        }
    } else {
        TrafficDescriptorComponent::OsIdOsAppId {
            os_id: [0u8; 16],
            os_app_id: descriptor.as_bytes().to_vec(),
        }
    };

    let mut components = Vec::new();
    for route in &rule.route_descriptors {
        if let Some(ref dnn) = route.dnn {
            components.push(RouteSelectionDescriptorComponent::Dnn(dnn.clone()));
        }
        if let Some(ref s_nssai) = route.s_nssai {
            components.push(RouteSelectionDescriptorComponent::SNssai {
                sst: s_nssai.sst,
                sd: s_nssai.sd,
            });
        }
        if let Some(mode) = route.ssc_mode {
            let mode = match mode {
                1 => SscMode::SscMode1,
                2 => SscMode::SscMode2,
                3 => SscMode::SscMode3,
                // TS 24.501 §9.11.4.16 defines modes 1-3 only. A configured 4
                // is a typo, not a mode, and defaulting it to 1 would apply a
                // continuity behaviour the operator did not ask for.
                other => {
                    debug!("URSP config: ignoring invalid SSC mode {other} (TS 24.501 §9.11.4.16)");
                    continue;
                }
            };
            components.push(RouteSelectionDescriptorComponent::SscMode(mode));
        }
        if let Some(session_type) = route.session_type {
            components.push(RouteSelectionDescriptorComponent::PduSessionType(
                match session_type {
                    ConfigPduSessionType::Ipv4 => PduSessionType::Ipv4,
                    ConfigPduSessionType::Ipv6 => PduSessionType::Ipv6,
                    ConfigPduSessionType::Ipv4v6 => PduSessionType::Ipv4v6,
                    ConfigPduSessionType::Unstructured => PduSessionType::Unstructured,
                    ConfigPduSessionType::Ethernet => PduSessionType::Ethernet,
                },
            ));
        }
    }
    if components.is_empty() {
        debug!(
            "URSP config: rule with precedence {} has no usable route selection descriptor; \
             dropping it so it cannot shadow a rule that can steer",
            rule.precedence
        );
        return None;
    }

    Some(UrspRule {
        precedence: rule.precedence,
        traffic_descriptor: vec![component],
        route_selection_descriptors: vec![RouteSelectionDescriptor {
            precedence: 1,
            components,
        }],
        ureri: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::RouteDescriptor;
    use nextgsim_common::types::SNssai;

    fn base_params() -> SmSessionParams {
        SmSessionParams {
            session_type: PduSessionTypeValue::Ipv4,
            ssc_mode: SscModeValue::SscMode1,
            dnn: Some("internet".to_string()),
            s_nssai: Some(vec![0x01]),
            emergency: false,
            requested_5qi: None,
            subscription_index: 0,
        }
    }

    fn config_rule(
        precedence: u8,
        descriptor: &str,
        dnn: Option<&str>,
        sst: Option<u8>,
    ) -> ConfigUrspRule {
        ConfigUrspRule {
            precedence,
            traffic_descriptor: descriptor.to_string(),
            route_descriptors: vec![RouteDescriptor {
                s_nssai: sst.map(SNssai::new),
                dnn: dnn.map(str::to_string),
                session_type: None,
                ssc_mode: None,
            }],
        }
    }

    fn policy(enabled: bool, rules: Vec<ConfigUrspRule>) -> UrspPolicy {
        let mut config = UeConfig::default();
        config.ursp_evaluation = enabled;
        config.ursp_rules = rules;
        UrspPolicy::from_config(&config)
    }

    /// #47 criterion 4a: two rules whose descriptors both match, and the LOWER
    /// precedence value wins (TS 24.526 §5.2 "increasing order of precedence").
    ///
    /// Both rules are match-all-by-app so both genuinely match; the rules are
    /// supplied in the WRONG order so a first-wins implementation that skips the
    /// sort fails.
    #[test]
    fn the_lower_precedence_value_wins_among_matching_rules() {
        let pol = policy(
            true,
            vec![
                config_rule(200, "com.example.app", Some("late"), Some(9)),
                config_rule(10, "com.example.app", Some("early"), Some(3)),
            ],
        );
        let app = ApplicationDescriptor::for_app_id(*b"com.example.app");
        let matched = pol.evaluate(&app).expect("both rules match");
        assert_eq!(matched.precedence, 10, "precedence 10 outranks 200");
        assert!(!matched.is_default);

        let params = matched.apply_to(&base_params());
        assert_eq!(params.dnn.as_deref(), Some("early"));
        assert_eq!(params.s_nssai, Some(vec![3]));
    }

    /// #47 criterion 4b: with no matching rule the DEFAULT (match-all) rule is
    /// applied.
    #[test]
    fn an_unmatched_application_falls_back_to_the_default_rule() {
        let pol = policy(
            true,
            vec![
                config_rule(10, "com.example.app", Some("specific"), Some(3)),
                config_rule(250, "*", Some("catch-all"), Some(1)),
            ],
        );
        let matched = pol
            .evaluate(&ApplicationDescriptor::for_app_id(*b"com.other.app"))
            .expect("the default rule must catch it");
        assert!(matched.is_default);
        assert_eq!(matched.precedence, 250);
        assert_eq!(
            matched.apply_to(&base_params()).dnn.as_deref(),
            Some("catch-all")
        );
    }

    /// #47: the default rule must NOT pre-empt a specific rule, even when its
    /// precedence value is lower.
    ///
    /// This is the subtle half of §5.2's "except the default URSP rule": a
    /// default rule at precedence 1 is still only a fallback. An implementation
    /// that merely sorts and takes the first match gets this wrong.
    #[test]
    fn a_low_precedence_default_rule_does_not_pre_empt_a_specific_rule() {
        let pol = policy(
            true,
            vec![
                config_rule(1, "*", Some("catch-all"), Some(1)),
                config_rule(200, "com.example.app", Some("specific"), Some(7)),
            ],
        );
        let matched = pol
            .evaluate(&ApplicationDescriptor::for_app_id(*b"com.example.app"))
            .expect("the specific rule must match");
        assert!(
            !matched.is_default,
            "the specific rule wins even at a worse precedence value"
        );
        assert_eq!(
            matched.apply_to(&base_params()).dnn.as_deref(),
            Some("specific")
        );
    }

    /// #47: with the switch OFF nothing is evaluated and the static parameters
    /// are returned unchanged.
    ///
    /// This is the default path, so it is the one that must not regress.
    #[test]
    fn evaluation_off_returns_the_static_parameters_unchanged() {
        let pol = policy(false, vec![config_rule(1, "*", Some("steered"), Some(9))]);
        assert!(!pol.is_enabled());
        let base = base_params();
        assert_eq!(
            pol.evaluate(&ApplicationDescriptor::for_dnn("internet")),
            None
        );
        assert_eq!(
            pol.resolve(&ApplicationDescriptor::for_dnn("internet"), &base),
            base,
            "the switch being off must be indistinguishable from having no rules"
        );
    }

    /// #47: a route that names only an S-NSSAI steers the slice and INHERITS the
    /// configured DNN.
    ///
    /// §5.2's route selection descriptor is an override set, not a whole config;
    /// clearing the unnamed fields would establish a session with no DNN.
    #[test]
    fn a_partial_route_overrides_only_what_it_names() {
        let pol = policy(true, vec![config_rule(10, "*", None, Some(4))]);
        let params = pol.resolve(&ApplicationDescriptor::for_dnn("internet"), &base_params());
        assert_eq!(params.s_nssai, Some(vec![4]), "the slice is steered");
        assert_eq!(
            params.dnn.as_deref(),
            Some("internet"),
            "the configured DNN survives a route that does not name one"
        );
        assert_eq!(params.session_type, PduSessionTypeValue::Ipv4);
        assert_eq!(params.ssc_mode, SscModeValue::SscMode1);
    }

    /// #47: the `dnn:` and `fqdn:` config spellings key on the right field, and a
    /// bare string is an App Id.
    #[test]
    fn the_config_traffic_descriptor_spellings_key_on_their_own_field() {
        let pol = policy(
            true,
            vec![
                config_rule(10, "dnn:ims", Some("ims-route"), None),
                config_rule(20, "fqdn:Example.COM", Some("fqdn-route"), None),
                config_rule(30, "com.example.app", Some("app-route"), None),
            ],
        );
        let dnn_match = pol
            .evaluate(&ApplicationDescriptor::for_dnn("ims"))
            .unwrap();
        assert_eq!(
            dnn_match.apply_to(&base_params()).dnn.as_deref(),
            Some("ims-route")
        );

        // FQDN comparison is case-insensitive, as DNS names are.
        let fqdn_app = ApplicationDescriptor {
            fqdn: Some("example.com".to_string()),
            ..Default::default()
        };
        assert_eq!(
            pol.evaluate(&fqdn_app)
                .unwrap()
                .apply_to(&base_params())
                .dnn
                .as_deref(),
            Some("fqdn-route")
        );

        let app_match = pol
            .evaluate(&ApplicationDescriptor::for_app_id(*b"com.example.app"))
            .unwrap();
        assert_eq!(
            app_match.apply_to(&base_params()).dnn.as_deref(),
            Some("app-route")
        );

        // And a DNN rule does not fire for an application that named no DNN.
        assert_eq!(
            pol.evaluate(&ApplicationDescriptor::for_app_id(*b"com.unknown")),
            None,
            "no rule matches and there is no default rule"
        );
    }

    /// #47: an application descriptor with NO information matches nothing.
    ///
    /// The opposite convention ("unknown matches anything") would have the
    /// highest-priority rule capture all traffic regardless of what it was
    /// written for.
    #[test]
    fn an_empty_application_descriptor_matches_no_specific_rule() {
        let pol = policy(
            true,
            vec![
                config_rule(10, "dnn:ims", Some("ims"), None),
                config_rule(20, "com.example.app", Some("app"), None),
            ],
        );
        assert_eq!(pol.evaluate(&ApplicationDescriptor::default()), None);
    }

    /// #47: a configured rule with no usable route is dropped, so it cannot
    /// shadow a later rule that can steer.
    #[test]
    fn a_config_rule_with_no_route_is_dropped_rather_than_kept_empty() {
        let routeless = ConfigUrspRule {
            precedence: 1,
            traffic_descriptor: "com.example.app".to_string(),
            route_descriptors: vec![RouteDescriptor {
                s_nssai: None,
                dnn: None,
                session_type: None,
                ssc_mode: None,
            }],
        };
        let pol = policy(
            true,
            vec![
                routeless,
                config_rule(50, "com.example.app", Some("usable"), None),
            ],
        );
        assert_eq!(pol.rule_count(), 1, "the routeless rule is not installed");
        let matched = pol
            .evaluate(&ApplicationDescriptor::for_app_id(*b"com.example.app"))
            .expect("the usable rule must be reachable");
        assert_eq!(matched.precedence, 50);
    }

    /// #47: an invalid configured SSC mode is ignored rather than defaulted.
    ///
    /// TS 24.501 §9.11.4.16 defines modes 1-3; a configured 4 is a typo, and
    /// defaulting it to mode 1 would apply a continuity behaviour the operator
    /// did not ask for.
    #[test]
    fn an_invalid_configured_ssc_mode_is_ignored_not_defaulted() {
        let rule = ConfigUrspRule {
            precedence: 10,
            traffic_descriptor: "*".to_string(),
            route_descriptors: vec![RouteDescriptor {
                s_nssai: Some(SNssai::new(5)),
                dnn: None,
                session_type: None,
                ssc_mode: Some(4),
            }],
        };
        let pol = policy(true, vec![rule]);
        let mut base = base_params();
        base.ssc_mode = SscModeValue::SscMode2;
        let params = pol.resolve(&ApplicationDescriptor::for_dnn("internet"), &base);
        assert_eq!(
            params.s_nssai,
            Some(vec![5]),
            "the valid part still applies"
        );
        assert_eq!(
            params.ssc_mode,
            SscModeValue::SscMode2,
            "the base SSC mode survives an invalid configured one"
        );
    }

    /// #47: network-delivered rules outrank configured ones at equal precedence.
    ///
    /// A PCF's policy is authoritative; a locally configured rule is a default.
    #[test]
    fn delivered_rules_outrank_configured_ones_at_equal_precedence() {
        let mut pol = policy(true, vec![config_rule(10, "*", Some("configured"), None)]);
        pol.set_delivered_rules(vec![UrspRule {
            precedence: 10,
            traffic_descriptor: vec![TrafficDescriptorComponent::MatchAll],
            route_selection_descriptors: vec![RouteSelectionDescriptor {
                precedence: 1,
                components: vec![RouteSelectionDescriptorComponent::Dnn(
                    "delivered".to_string(),
                )],
            }],
            ureri: None,
        }]);
        assert_eq!(pol.rule_count(), 2);
        assert_eq!(
            pol.resolve(&ApplicationDescriptor::for_dnn("internet"), &base_params())
                .dnn
                .as_deref(),
            Some("delivered")
        );
    }

    /// #47: the lowest-precedence route selection descriptor of a matched rule is
    /// the one selected (§5.2 walks RSDs in increasing precedence too).
    #[test]
    fn the_lowest_precedence_route_of_a_matched_rule_is_selected() {
        let mut pol = policy(true, vec![]);
        pol.set_delivered_rules(vec![UrspRule {
            precedence: 5,
            traffic_descriptor: vec![TrafficDescriptorComponent::MatchAll],
            route_selection_descriptors: vec![
                RouteSelectionDescriptor {
                    precedence: 200,
                    components: vec![RouteSelectionDescriptorComponent::Dnn("worse".to_string())],
                },
                RouteSelectionDescriptor {
                    precedence: 3,
                    components: vec![RouteSelectionDescriptorComponent::Dnn("better".to_string())],
                },
            ],
            ureri: None,
        }]);
        assert_eq!(
            pol.resolve(&ApplicationDescriptor::for_dnn("internet"), &base_params())
                .dnn
                .as_deref(),
            Some("better")
        );
    }

    /// #47: all components of a traffic descriptor must match (conjunction).
    ///
    /// A rule scoped to "this app on this port" must not fire for the app on
    /// another port.
    #[test]
    fn every_traffic_descriptor_component_must_match() {
        let mut pol = policy(true, vec![]);
        pol.set_delivered_rules(vec![UrspRule {
            precedence: 10,
            traffic_descriptor: vec![
                TrafficDescriptorComponent::Dnn("ims".to_string()),
                TrafficDescriptorComponent::SingleRemotePort(5060),
            ],
            route_selection_descriptors: vec![RouteSelectionDescriptor {
                precedence: 1,
                components: vec![RouteSelectionDescriptorComponent::Dnn("sip".to_string())],
            }],
            ureri: None,
        }]);

        let both = ApplicationDescriptor {
            dnn: Some("ims".to_string()),
            port: Some(5060),
            ..Default::default()
        };
        assert!(pol.evaluate(&both).is_some(), "both components match");

        let wrong_port = ApplicationDescriptor {
            dnn: Some("ims".to_string()),
            port: Some(80),
            ..Default::default()
        };
        assert_eq!(
            pol.evaluate(&wrong_port),
            None,
            "one component matching is not a match"
        );

        let dnn_only = ApplicationDescriptor::for_dnn("ims");
        assert_eq!(
            pol.evaluate(&dnn_only),
            None,
            "an unknown port cannot satisfy a port component"
        );
    }

    /// #47: the IPv4 mask and IPv6 prefix are honoured, including a prefix that
    /// does not fall on an octet boundary.
    #[test]
    fn address_components_honour_their_mask_and_prefix() {
        let mut pol = policy(true, vec![]);
        pol.set_delivered_rules(vec![UrspRule {
            precedence: 10,
            traffic_descriptor: vec![TrafficDescriptorComponent::Ipv4RemoteAddress {
                addr: [10, 45, 0, 0],
                mask: [255, 255, 0, 0],
            }],
            route_selection_descriptors: vec![RouteSelectionDescriptor {
                precedence: 1,
                components: vec![RouteSelectionDescriptorComponent::Dnn("v4".to_string())],
            }],
            ureri: None,
        }]);
        let inside = ApplicationDescriptor {
            ipv4: Some([10, 45, 7, 9]),
            ..Default::default()
        };
        let outside = ApplicationDescriptor {
            ipv4: Some([10, 46, 7, 9]),
            ..Default::default()
        };
        assert!(pol.evaluate(&inside).is_some());
        assert_eq!(pol.evaluate(&outside), None);

        // /12 is a non-octet-aligned prefix, which is where an implementation
        // that only compares whole octets goes wrong.
        let mut v6 = [0u8; 16];
        v6[0] = 0x20;
        v6[1] = 0x01;
        pol.set_delivered_rules(vec![UrspRule {
            precedence: 10,
            traffic_descriptor: vec![TrafficDescriptorComponent::Ipv6RemoteAddress {
                addr: v6,
                prefix_len: 12,
            }],
            route_selection_descriptors: vec![RouteSelectionDescriptor {
                precedence: 1,
                components: vec![RouteSelectionDescriptorComponent::Dnn("v6".to_string())],
            }],
            ureri: None,
        }]);
        // 2001::/12 pins byte 0 = 0x20 and the HIGH NIBBLE of byte 1 = 0x0.
        // `[0x20, 0x0F]` is inside it and `[0x20, 0xFF]` is not, which is the pair
        // that distinguishes a real sub-octet comparison from one that compares
        // whole octets (the latter rejects 0x0F because it is not 0x01).
        let mut in_prefix = [0u8; 16];
        in_prefix[0] = 0x20;
        in_prefix[1] = 0x0F;
        let mut out_of_prefix = [0u8; 16];
        out_of_prefix[0] = 0x20;
        out_of_prefix[1] = 0xFF;
        assert!(pol
            .evaluate(&ApplicationDescriptor {
                ipv6: Some(in_prefix),
                ..Default::default()
            })
            .is_some());
        assert_eq!(
            pol.evaluate(&ApplicationDescriptor {
                ipv6: Some(out_of_prefix),
                ..Default::default()
            }),
            None
        );
    }

    // --- Traffic classification (issue #97) ---

    /// A minimal IPv4 + TCP packet to a given destination and port.
    fn ipv4_tcp(dst: [u8; 4], dst_port: u16) -> Vec<u8> {
        let mut p = vec![0u8; 40];
        p[0] = 0x45; // version 4, IHL 5 words
        p[9] = 6; // TCP
        p[12..16].copy_from_slice(&[10, 0, 0, 2]); // source
        p[16..20].copy_from_slice(&dst);
        p[20..22].copy_from_slice(&1234u16.to_be_bytes()); // source port
        p[22..24].copy_from_slice(&dst_port.to_be_bytes());
        p
    }

    #[test]
    fn an_ipv4_packet_classifies_to_its_remote_address_protocol_and_port() {
        let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([93, 184, 216, 34], 443))
            .expect("a classifiable packet");
        assert_eq!(app.ipv4, Some([93, 184, 216, 34]));
        assert_eq!(app.protocol, Some(6));
        assert_eq!(app.port, Some(443));
        // Not derivable from a packet, and therefore absent rather than guessed.
        assert_eq!(app.ipv6, None);
        assert_eq!(app.fqdn, None);
        assert_eq!(app.os_app_id, None);
    }

    #[test]
    fn the_remote_address_is_the_destination_not_the_source() {
        // A traffic descriptor names the far end of an uplink packet. Reading the
        // source would match every rule against the UE's own address.
        let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([203, 0, 113, 5], 80))
            .expect("classifiable");
        assert_eq!(app.ipv4, Some([203, 0, 113, 5]));
        assert_ne!(app.ipv4, Some([10, 0, 0, 2]), "that is the SOURCE address");
        assert_eq!(app.port, Some(80));
        assert_ne!(app.port, Some(1234), "that is the SOURCE port");
    }

    #[test]
    fn an_ipv4_packet_with_options_reads_the_transport_header_at_the_right_offset() {
        // IHL 6 words = 24 octets, so the ports start at 24 and not 20. Assuming a
        // 20-octet header would read the option bytes as a port.
        let mut p = ipv4_tcp([198, 51, 100, 1], 0);
        p[0] = 0x46; // IHL 6
        p[20..24].copy_from_slice(&[0x01, 0x01, 0x01, 0x00]); // options (NOPs)
        p[24..26].copy_from_slice(&5555u16.to_be_bytes()); // source port
        p[26..28].copy_from_slice(&8443u16.to_be_bytes()); // destination port

        let app = ApplicationDescriptor::from_uplink_packet(&p).expect("classifiable");
        assert_eq!(app.port, Some(8443));
    }

    #[test]
    fn an_ipv6_packet_classifies_to_its_remote_address_and_port() {
        let mut p = vec![0u8; 60];
        p[0] = 0x60; // version 6
        p[6] = 17; // UDP
        let dst = [0x20u8, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        p[24..40].copy_from_slice(&dst);
        p[40..42].copy_from_slice(&9000u16.to_be_bytes());
        p[42..44].copy_from_slice(&53u16.to_be_bytes());

        let app = ApplicationDescriptor::from_uplink_packet(&p).expect("classifiable");
        assert_eq!(app.ipv6, Some(dst));
        assert_eq!(app.ipv4, None);
        assert_eq!(app.protocol, Some(17));
        assert_eq!(app.port, Some(53));
    }

    #[test]
    fn a_non_port_protocol_classifies_with_a_protocol_and_no_port() {
        // ICMP has no ports. A protocol-keyed rule must still match.
        let mut p = ipv4_tcp([192, 0, 2, 1], 0);
        p[9] = 1; // ICMP
        let app = ApplicationDescriptor::from_uplink_packet(&p).expect("classifiable");
        assert_eq!(app.protocol, Some(1));
        assert_eq!(app.port, None, "ICMP has no port to report");
    }

    #[test]
    fn an_unclassifiable_packet_yields_nothing_rather_than_a_partial_descriptor() {
        // A descriptor with a plausible address and a garbage port would match a
        // rule the flow does not belong to.
        assert!(ApplicationDescriptor::from_uplink_packet(&[]).is_none());
        assert!(
            ApplicationDescriptor::from_uplink_packet(&[0x45]).is_none(),
            "truncated"
        );
        // An IHL below the 5-word minimum is malformed.
        let mut bad_ihl = ipv4_tcp([192, 0, 2, 1], 80);
        bad_ihl[0] = 0x44;
        assert!(ApplicationDescriptor::from_uplink_packet(&bad_ihl).is_none());
        // A TCP packet whose transport header is cut short.
        let truncated = ipv4_tcp([192, 0, 2, 1], 80)[..22].to_vec();
        assert!(ApplicationDescriptor::from_uplink_packet(&truncated).is_none());
        // Neither IPv4 nor IPv6.
        assert!(ApplicationDescriptor::from_uplink_packet(&[0x75; 40]).is_none());
        // An IPv6 packet shorter than its fixed header.
        assert!(ApplicationDescriptor::from_uplink_packet(&[0x60; 30]).is_none());
    }

    #[test]
    fn a_classified_flow_matches_the_address_protocol_and_port_components() {
        // The point of the whole change: components that could never match before
        // now do, because production can finally report a flow.
        let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([93, 184, 216, 34], 443))
            .expect("classifiable");

        assert!(
            app.matches_component(&TrafficDescriptorComponent::Ipv4RemoteAddress {
                addr: [93, 184, 216, 34],
                mask: [255, 255, 255, 255],
            })
        );
        assert!(app.matches_component(&TrafficDescriptorComponent::ProtocolIdentifier(6)));
        assert!(app.matches_component(&TrafficDescriptorComponent::SingleRemotePort(443)));
        assert!(
            app.matches_component(&TrafficDescriptorComponent::RemotePortRange {
                low: 440,
                high: 450,
            })
        );
        // And a rule for a different flow still does not match.
        assert!(!app.matches_component(&TrafficDescriptorComponent::SingleRemotePort(80)));
    }

    #[test]
    fn the_configured_os_identity_reaches_the_descriptor() {
        let mut config = nextgsim_common::config::UeConfig::default();
        config.ursp_os_app_id = Some("com.example.video".to_string());
        config.ursp_os_id = Some("0123456789abcdef0123456789abcdef".to_string());

        let (os_id, os_app_id) = ApplicationDescriptor::configured_os_identity(&config);
        assert_eq!(os_app_id.as_deref(), Some(&b"com.example.video"[..]));
        assert_eq!(os_id.map(|id| id[0]), Some(0x01));
        assert_eq!(os_id.map(|id| id[15]), Some(0xef));

        let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([1, 1, 1, 1], 443))
            .expect("classifiable")
            .with_os_identity(os_id, os_app_id.clone());
        assert!(
            app.matches_component(&TrafficDescriptorComponent::OsIdOsAppId {
                os_id: os_id.expect("a valid configured OS Id"),
                os_app_id: os_app_id.clone().unwrap(),
            })
        );
    }

    #[test]
    fn a_malformed_os_id_is_dropped_rather_than_truncated() {
        // Half a UUID identifies a different OS.
        let mut config = nextgsim_common::config::UeConfig::default();
        config.ursp_os_id = Some("0123456789abcdef".to_string());
        let (os_id, _) = ApplicationDescriptor::configured_os_identity(&config);
        assert!(os_id.is_none());

        // A dashed UUID is accepted, because that is how a UUID is usually written.
        config.ursp_os_id = Some("01234567-89ab-cdef-0123-456789abcdef".to_string());
        let (os_id, _) = ApplicationDescriptor::configured_os_identity(&config);
        assert_eq!(os_id.map(|id| id[0]), Some(0x01));

        // Non-hex is rejected, not read as zeroes.
        config.ursp_os_id = Some("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz".to_string());
        let (os_id, _) = ApplicationDescriptor::configured_os_identity(&config);
        assert!(os_id.is_none());
    }

    #[test]
    fn no_configured_os_identity_leaves_the_components_unmatchable() {
        // The honest default: without a declared app identity, an OsIdOsAppId rule
        // cannot fire, and this asserts it rather than leaving it implied.
        let config = nextgsim_common::config::UeConfig::default();
        let (os_id, os_app_id) = ApplicationDescriptor::configured_os_identity(&config);
        assert!(os_id.is_none());
        assert!(os_app_id.is_none());

        let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([1, 1, 1, 1], 443))
            .expect("classifiable")
            .with_os_identity(os_id, os_app_id);
        assert!(
            !app.matches_component(&TrafficDescriptorComponent::OsIdOsAppId {
                os_id: [0u8; 16],
                os_app_id: b"com.example.video".to_vec(),
            })
        );
    }

    // --- Flow-keyed config descriptor spellings (issue #97) ---

    fn rule_with(descriptor: &str) -> ConfigUrspRule {
        use nextgsim_common::config::RouteDescriptor;
        ConfigUrspRule {
            precedence: 10,
            traffic_descriptor: descriptor.to_string(),
            route_descriptors: vec![RouteDescriptor {
                s_nssai: None,
                dnn: Some("video".to_string()),
                session_type: None,
                ssc_mode: None,
            }],
        }
    }

    #[test]
    fn a_configured_port_rule_matches_a_classified_flow() {
        // The whole chain in one assertion: an operator writes `port:443`, the
        // classifier reads 443 off a packet, and the rule fires. Before issue #97
        // neither half existed.
        let rule = convert_config_rule(&rule_with("port:443")).expect("a usable rule");
        let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([93, 184, 216, 34], 443))
            .expect("classifiable");
        assert!(app.matches(&rule.traffic_descriptor));

        let other = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([93, 184, 216, 34], 80))
            .expect("classifiable");
        assert!(!other.matches(&rule.traffic_descriptor));
    }

    #[test]
    fn a_configured_port_range_matches_its_bounds_inclusively() {
        let rule = convert_config_rule(&rule_with("port:8000-8100")).expect("usable");
        for (port, expected) in [
            (7999u16, false),
            (8000, true),
            (8050, true),
            (8100, true),
            (8101, false),
        ] {
            let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([1, 1, 1, 1], port))
                .expect("classifiable");
            assert_eq!(
                app.matches(&rule.traffic_descriptor),
                expected,
                "port {port}"
            );
        }
    }

    #[test]
    fn a_reversed_port_range_is_rejected_rather_than_swapped() {
        // Reading `500-100` as 100-500 applies a policy nobody wrote.
        assert!(convert_config_rule(&rule_with("port:500-100")).is_none());
        assert!(convert_config_rule(&rule_with("port:not-a-port")).is_none());
        assert!(convert_config_rule(&rule_with("port:70000")).is_none());
    }

    #[test]
    fn a_configured_protocol_rule_matches_the_classified_protocol() {
        let rule = convert_config_rule(&rule_with("proto:6")).expect("usable");
        let tcp = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([1, 1, 1, 1], 443))
            .expect("classifiable");
        assert!(tcp.matches(&rule.traffic_descriptor));

        let mut udp_packet = ipv4_tcp([1, 1, 1, 1], 443);
        udp_packet[9] = 17;
        let udp = ApplicationDescriptor::from_uplink_packet(&udp_packet).expect("classifiable");
        assert!(!udp.matches(&rule.traffic_descriptor));

        assert!(convert_config_rule(&rule_with("proto:300")).is_none());
    }

    #[test]
    fn a_bare_configured_ipv4_address_is_an_exact_match_not_a_wildcard() {
        // Defaulting a missing prefix to 0 would turn "this host" into "every
        // host", which is the difference between a rule and a catch-all.
        let rule = convert_config_rule(&rule_with("ipv4:93.184.216.34")).expect("usable");
        let exact = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([93, 184, 216, 34], 443))
            .expect("classifiable");
        assert!(exact.matches(&rule.traffic_descriptor));
        let neighbour =
            ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([93, 184, 216, 35], 443))
                .expect("classifiable");
        assert!(!neighbour.matches(&rule.traffic_descriptor));
    }

    #[test]
    fn a_configured_ipv4_prefix_matches_the_whole_subnet() {
        let rule = convert_config_rule(&rule_with("ipv4:10.20.0.0/16")).expect("usable");
        for (addr, expected) in [
            ([10u8, 20, 0, 1], true),
            ([10, 20, 255, 254], true),
            ([10, 21, 0, 1], false),
        ] {
            let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp(addr, 443))
                .expect("classifiable");
            assert_eq!(app.matches(&rule.traffic_descriptor), expected, "{addr:?}");
        }
        // A /0 matches everything, which an operator may legitimately want -- but
        // only when they write it.
        let any = convert_config_rule(&rule_with("ipv4:0.0.0.0/0")).expect("usable");
        let app = ApplicationDescriptor::from_uplink_packet(&ipv4_tcp([203, 0, 113, 9], 443))
            .expect("classifiable");
        assert!(app.matches(&any.traffic_descriptor));
    }

    #[test]
    fn a_malformed_configured_address_drops_the_rule_rather_than_widening_it() {
        // A rule that fell back to match-all would steer every flow.
        for descriptor in [
            "ipv4:999.1.1.1",
            "ipv4:10.0.0",
            "ipv4:10.0.0.1/33",
            "ipv6:not:an:address",
            "ipv6:2001:db8::1/129",
            "ipv6:2001::db8::1",
        ] {
            assert!(
                convert_config_rule(&rule_with(descriptor)).is_none(),
                "{descriptor} must drop the rule"
            );
        }
    }

    #[test]
    fn a_configured_ipv6_prefix_matches_a_classified_ipv6_flow() {
        let rule = convert_config_rule(&rule_with("ipv6:2001:db8::/32")).expect("usable");

        let mut packet = vec![0u8; 60];
        packet[0] = 0x60;
        packet[6] = 6;
        packet[24..40].copy_from_slice(&[
            0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x99,
        ]);
        packet[42..44].copy_from_slice(&443u16.to_be_bytes());
        let app = ApplicationDescriptor::from_uplink_packet(&packet).expect("classifiable");
        assert!(app.matches(&rule.traffic_descriptor));

        // A different /32 must not match.
        packet[24..28].copy_from_slice(&[0x20, 0x02, 0x0d, 0xb8]);
        let other = ApplicationDescriptor::from_uplink_packet(&packet).expect("classifiable");
        assert!(!other.matches(&rule.traffic_descriptor));
    }

    #[test]
    fn the_ipv6_parser_expands_a_double_colon_run() {
        assert_eq!(
            parse_ipv6_address("2001:db8::1"),
            Some([0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        );
        assert_eq!(parse_ipv6_address("::1").map(|a| a[15]), Some(1));
        assert_eq!(parse_ipv6_address("::"), Some([0u8; 16]));
        // A full eight-group form needs all eight.
        assert!(parse_ipv6_address("2001:db8:0:0:0:0:0").is_none());
        assert_eq!(
            parse_ipv6_address("2001:0db8:0000:0000:0000:0000:0000:0001"),
            parse_ipv6_address("2001:db8::1")
        );
    }
}
