//! Live-state knowledge exposure: one question, one bundle (nextgsim#488).
//!
//! The rest of this crate is a general knowledge graph. This module is the
//! narrow, live slice the d447a-nkef-live demo needs: facts **observed during
//! this run** from three domains, each tagged with the domain, the producer that
//! reported it and when, and a bundle answer that joins them.
//!
//! - **Domains.** `ran` (this gNB's RLS: cell load, UE measurements), `n2`
//!   (this gNB's NGAP: the AMF name, GUAMI, AMF UE NGAP ID) and `sba` (the core's
//!   own answers: an amfd LOCATION_REPORT / CONNECTIVITY_STATE_REPORT, and NRF
//!   discovery).
//! - **Freshness and expiry** are project-defined. Every fact carries
//!   `observed_at`; a fact whose producer stopped reporting it gets
//!   `expired_at` and is kept, so a consumer sees it went away rather than
//!   finding it silently missing.
//! - **The join.** The UE's path is `ue>cell>gnb>amf>nf-instance`: the UE's
//!   serving cell, this gNB, the AMF it reached over N2, and the NRF NF instance
//!   whose `amfInfo.guamiList` holds that AMF's GUAMI.
//! - **Consumers.** [`Consumer::Internal`] sees every fact. [`Consumer::External`]
//!   gets the TS 23.502 §4.15.7 shape only: per-area congestion, with no
//!   location and no identity ("the NSR shall not include any 3GPP location
//!   information"). The policy filters **before** ranking, so a fact an external
//!   consumer may not see can never take a top-k slot or spend budget.
//! - **A hard budget.** A bundle never exceeds its token budget, not even by its
//!   first fact.
//!
//! Ranking is lexical overlap over a handful of facts. This is not natural
//! language Q&A and not retrieval research: it exists so one question returns
//! one bounded bundle.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// A fact's network domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Domain {
    /// This gNB's radio side (RLS).
    Ran,
    /// This gNB's N2 (NGAP) state.
    N2,
    /// The core's service-based interface (amfd, nrfd).
    Sba,
}

impl Domain {
    /// The domain's label in a bundle.
    pub fn as_str(self) -> &'static str {
        match self {
            Domain::Ran => "ran",
            Domain::N2 => "n2",
            Domain::Sba => "sba",
        }
    }
}

/// Who asks: the policy is applied inside search, before top-k.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Consumer {
    /// An operator-side consumer: every fact.
    Internal,
    /// A third party: per-area congestion only (TS 23.502 §4.15.7).
    External,
}

impl Consumer {
    /// Parse `internal` / `external`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "internal" => Some(Self::Internal),
            "external" => Some(Self::External),
            _ => None,
        }
    }

    /// The consumer's label in a bundle.
    pub fn as_str(self) -> &'static str {
        match self {
            Consumer::Internal => "internal",
            Consumer::External => "external",
        }
    }
}

/// What a fact is about. The join reads these, so they are typed rather than
/// inferred from the subject string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FactKind {
    /// A cell's measured load (ran).
    CellLoad,
    /// A UE as the radio side sees it: serving cell, measurement (ran).
    UeRadio,
    /// This gNB's own identity (n2).
    Gnb,
    /// An AMF this gNB set up N2 with: name, GUAMI (n2).
    Amf,
    /// A UE's N2 association: AMF UE NGAP ID (n2).
    UeN2,
    /// The AMF's last known location of the UE (sba, LOCATION_REPORT).
    UeLocation,
    /// The AMF's CM state of the UE (sba, CONNECTIVITY_STATE_REPORT).
    UeConnectivity,
    /// An NF instance the NRF discovered (sba).
    NfInstance,
}

/// One observed fact.
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    /// Its domain.
    pub domain: Domain,
    /// What it is about.
    pub kind: FactKind,
    /// Who reported it, e.g. `gnb-rls`, `gnb-ngap`, `amf`, `nrf`.
    pub producer: String,
    /// The entity, unique within its kind, e.g. `cell-16`, `nf-<uuid>`.
    pub subject: String,
    /// Its attributes.
    pub attrs: BTreeMap<String, String>,
    /// When it was observed, ms since the Unix epoch.
    pub observed_at_ms: u64,
    /// When its producer stopped reporting it, if it did.
    pub expired_at_ms: Option<u64>,
}

impl Fact {
    /// A fact observed at `observed_at_ms`.
    pub fn new(
        domain: Domain,
        kind: FactKind,
        producer: impl Into<String>,
        subject: impl Into<String>,
        observed_at_ms: u64,
    ) -> Self {
        Self {
            domain,
            kind,
            producer: producer.into(),
            subject: subject.into(),
            attrs: BTreeMap::new(),
            observed_at_ms,
            expired_at_ms: None,
        }
    }

    /// With attribute `k=v`.
    pub fn with(mut self, k: &str, v: impl ToString) -> Self {
        self.attrs.insert(k.to_string(), v.to_string());
        self
    }

    /// Attribute `k`.
    pub fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.get(k).map(String::as_str)
    }

    /// Whether its producer still reports it.
    pub fn is_live(&self) -> bool {
        self.expired_at_ms.is_none()
    }

    fn text(&self) -> String {
        let mut t = format!(
            "{} {} {} ",
            self.domain.as_str(),
            self.producer,
            self.subject
        );
        for (k, v) in &self.attrs {
            let _ = write!(t, "{k} {v} ");
        }
        t
    }

    fn render(&self, now_ms: u64) -> String {
        let mut line = format!(
            "fact domain={} producer={} subject={} age={}s",
            self.domain.as_str(),
            self.producer,
            self.subject,
            now_ms.saturating_sub(self.observed_at_ms) / 1000
        );
        for (k, v) in &self.attrs {
            let _ = write!(line, " {k}={v}");
        }
        if let Some(at) = self.expired_at_ms {
            let _ = write!(line, " expired_at={}", rfc3339_ms(at));
        }
        line
    }
}

/// The project-defined load rule behind an external consumer's congestion level
/// (TS 23.502 §4.15.7 NOTE 1 allows High/Medium/Low; there the NEF derives it
/// from NWDAF User Data Congestion analytics, here it comes from the cell's own
/// measured throughput, a labelled departure).
pub fn congestion_level(throughput_mbps: f32) -> &'static str {
    if throughput_mbps < LOW_BELOW_MBPS {
        "Low"
    } else if throughput_mbps < MEDIUM_BELOW_MBPS {
        "Medium"
    } else {
        "High"
    }
}

/// Below this cell throughput the level is `Low`.
pub const LOW_BELOW_MBPS: f32 = 0.05;
/// Below this (and not `Low`) it is `Medium`; above, `High`.
pub const MEDIUM_BELOW_MBPS: f32 = 5.0;

/// Tokens a text costs, estimated as whitespace-separated words. Deterministic
/// and conservative enough for a hard cap on a short bundle.
pub fn token_cost(text: &str) -> usize {
    text.split_whitespace().count()
}

/// The answer to one question.
#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    /// The rendered bundle: a header line, then one line per fact.
    pub text: String,
    /// Tokens it spends, never more than its budget.
    pub tokens: usize,
    /// Facts it carries.
    pub facts: usize,
}

/// The live facts of one run.
#[derive(Debug, Clone)]
pub struct LiveStore {
    facts: BTreeMap<(FactKind, String), Fact>,
    started_at_ms: u64,
    /// This gNB's NR cell identity, which a consistent AMF location names.
    serving_nci: Option<u64>,
}

impl LiveStore {
    /// A store for a run that started at `started_at_ms`.
    pub fn new(started_at_ms: u64) -> Self {
        Self {
            facts: BTreeMap::new(),
            started_at_ms,
            serving_nci: None,
        }
    }

    /// When the run started (`NKEF task started at=`).
    pub fn started_at_ms(&self) -> u64 {
        self.started_at_ms
    }

    /// Name this gNB's NCI, for the location consistency check.
    pub fn set_serving_nci(&mut self, nci: u64) {
        self.serving_nci = Some(nci);
    }

    /// Record `fact`, replacing the previous one of its kind and subject. A
    /// refreshed fact is live again.
    pub fn upsert(&mut self, fact: Fact) {
        self.facts.insert((fact.kind, fact.subject.clone()), fact);
    }

    /// Every fact, live or expired.
    pub fn facts(&self) -> impl Iterator<Item = &Fact> {
        self.facts.values()
    }

    /// The fact of `kind` about `subject`.
    pub fn get(&self, kind: FactKind, subject: &str) -> Option<&Fact> {
        self.facts.get(&(kind, subject.to_string()))
    }

    /// Facts of `kind` from `producer` whose subject is not in `present` get
    /// `expired_at = at_ms` (once: an already expired fact keeps its time). The
    /// NRF poll calls this with the instances discovery still returns. Returns
    /// the subjects that expired now.
    pub fn expire_missing(
        &mut self,
        kind: FactKind,
        producer: &str,
        present: &BTreeSet<String>,
        at_ms: u64,
    ) -> Vec<String> {
        let mut gone = Vec::new();
        for ((k, subject), fact) in &mut self.facts {
            if *k == kind
                && fact.producer == producer
                && fact.is_live()
                && !present.contains(subject)
            {
                fact.expired_at_ms = Some(at_ms);
                gone.push(subject.clone());
            }
        }
        gone
    }

    /// The domains with at least one live fact.
    pub fn live_domains(&self) -> BTreeSet<Domain> {
        self.facts
            .values()
            .filter(|f| f.is_live())
            .map(|f| f.domain)
            .collect()
    }

    fn first_live(&self, kind: FactKind) -> Option<&Fact> {
        self.facts
            .values()
            .filter(|f| f.kind == kind && f.is_live())
            .max_by_key(|f| f.observed_at_ms)
    }

    /// The join `ue>cell>gnb>amf>nf-instance`, as far as it holds, and the NRF
    /// NF instance it ends at.
    pub fn path(&self) -> (Vec<&'static str>, Option<String>) {
        let mut hops = Vec::new();
        let Some(ue) = self.first_live(FactKind::UeRadio) else {
            return (hops, None);
        };
        hops.push("ue");
        let cell = ue
            .attr("cell")
            .and_then(|c| self.get(FactKind::CellLoad, &format!("cell-{c}")))
            .filter(|f| f.is_live());
        if cell.is_none() {
            return (hops, None);
        }
        hops.push("cell");
        let Some(_gnb) = self.first_live(FactKind::Gnb) else {
            return (hops, None);
        };
        hops.push("gnb");
        // The AMF the UE's N2 association runs to, else the one AMF set up.
        let amf = self
            .first_live(FactKind::UeN2)
            .and_then(|n2| n2.attr("amf"))
            .and_then(|a| self.get(FactKind::Amf, a))
            .filter(|f| f.is_live())
            .or_else(|| self.first_live(FactKind::Amf));
        let Some(amf) = amf else {
            return (hops, None);
        };
        hops.push("amf");
        let Some(guami) = amf.attr("guami") else {
            return (hops, None);
        };
        let nf = self.facts.values().find(|f| {
            f.kind == FactKind::NfInstance
                && f.is_live()
                && f.attr("nfType") == Some("AMF")
                && f.attr("guamis")
                    .is_some_and(|g| g.split(',').any(|x| x == guami))
        });
        match nf {
            Some(nf) => {
                hops.push("nf-instance");
                (hops, nf.attr("nfInstanceId").map(str::to_string))
            }
            None => (hops, None),
        }
    }

    /// Whether the AMF's location of the UE agrees with this gNB: `true` when
    /// it names this gNB's cell and carries a `ueLocationTimestamp`, `false`
    /// when it names another cell, `unknown` without a timestamped location.
    ///
    /// The echo caveat: the AMF's NCGI is the ULI this gNB sent, so `true`
    /// proves a round trip through the core, not an independent fix.
    pub fn consistent(&self) -> &'static str {
        let Some(loc) = self.first_live(FactKind::UeLocation) else {
            return "unknown";
        };
        if loc.attr("ueLocationTimestamp").is_none() {
            return "unknown";
        }
        let (Some(nci), Some(cell)) = (
            self.serving_nci,
            loc.attr("nrCellId")
                .and_then(|c| u64::from_str_radix(c, 16).ok()),
        ) else {
            return "unknown";
        };
        if nci == cell {
            "true"
        } else {
            "false"
        }
    }

    /// Whether the AMF's location predates this run (a gNB restarted without
    /// the UE): `true`, `false`, or `unknown` without a timestamped location.
    pub fn stale(&self) -> &'static str {
        let Some(ts) = self
            .first_live(FactKind::UeLocation)
            .and_then(|l| l.attr("ueLocationTimestampEpoch"))
            .and_then(|s| s.parse::<u64>().ok())
        else {
            return "unknown";
        };
        if ts.saturating_mul(1000) < self.started_at_ms.saturating_sub(999) {
            "true"
        } else {
            "false"
        }
    }

    /// The UE's CM state as the AMF reported it.
    pub fn cm(&self) -> String {
        self.first_live(FactKind::UeConnectivity)
            .and_then(|f| f.attr("cmState"))
            .unwrap_or("unknown")
            .to_string()
    }

    /// Answer `question` for `consumer` within `budget` tokens, at `now_ms`.
    pub fn bundle(&self, question: &str, consumer: Consumer, budget: usize, now_ms: u64) -> Bundle {
        match consumer {
            Consumer::Internal => self.internal_bundle(question, budget, now_ms),
            Consumer::External => self.external_bundle(budget, now_ms),
        }
    }

    fn internal_bundle(&self, question: &str, budget: usize, now_ms: u64) -> Bundle {
        let domains: Vec<&str> = self.live_domains().iter().map(|d| d.as_str()).collect();
        let (hops, nf) = self.path();
        let mut header = format!(
            "NKEF bundle consumer=internal domains={} consistent={} path={} cm={} stale={}",
            if domains.is_empty() {
                "none".to_string()
            } else {
                domains.join(",")
            },
            self.consistent(),
            if hops.is_empty() {
                "none".to_string()
            } else {
                hops.join(">")
            },
            self.cm(),
            self.stale(),
        );
        if let Some(nf) = nf {
            let _ = write!(header, " amf_nf_instance={nf}");
        }
        // Rank by lexical overlap with the question, then freshness.
        let words: BTreeSet<String> = question
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(str::to_lowercase)
            .collect();
        let mut ranked: Vec<(usize, &Fact)> = self
            .facts
            .values()
            .map(|f| {
                let text = f.text().to_lowercase();
                let score = words.iter().filter(|w| text.contains(w.as_str())).count();
                (score, f)
            })
            .collect();
        ranked.sort_by(|(a, fa), (b, fb)| b.cmp(a).then(fb.observed_at_ms.cmp(&fa.observed_at_ms)));
        let lines: Vec<String> = ranked.iter().map(|(_, f)| f.render(now_ms)).collect();
        assemble(header, lines, budget)
    }

    fn external_bundle(&self, budget: usize, now_ms: u64) -> Bundle {
        // The policy, applied before anything is ranked or counted: cells only,
        // live only, projected to an area index and a level. The area number is
        // an ordinal in this bundle, not the cell identity.
        let lines: Vec<String> = self
            .facts
            .values()
            .filter(|f| f.kind == FactKind::CellLoad && f.is_live())
            .enumerate()
            .map(|(i, f)| {
                let mbps = f
                    .attr("throughput_mbps")
                    .and_then(|t| t.parse::<f32>().ok())
                    .unwrap_or(0.0);
                format!(
                    "area-{} congestion={} age={}s",
                    i + 1,
                    congestion_level(mbps),
                    now_ms.saturating_sub(f.observed_at_ms) / 1000
                )
            })
            .collect();
        let header = format!("NKEF bundle consumer=external areas={}", lines.len());
        assemble(header, lines, budget)
    }
}

/// Header plus as many lines as fit in `budget`, in order. The header is always
/// counted, and a line that does not fit is not admitted, first one included.
fn assemble(header: String, lines: Vec<String>, budget: usize) -> Bundle {
    // Room for the trailer `tokens=N/B`.
    let trailer = 1;
    let mut tokens = token_cost(&header) + trailer;
    let mut text = header;
    let mut facts = 0;
    if tokens > budget {
        // Not even the header fits: an empty bundle, which the caller reports.
        return Bundle {
            text: String::new(),
            tokens: 0,
            facts: 0,
        };
    }
    for line in lines {
        let cost = token_cost(&line);
        if tokens + cost > budget {
            break;
        }
        tokens += cost;
        text.push('\n');
        text.push_str(&line);
        facts += 1;
    }
    // The header's trailer token, appended last so it reports the true total.
    if let Some(end) = text.find('\n') {
        text.insert_str(end, &format!(" tokens={tokens}/{budget}"));
    } else {
        let _ = write!(text, " tokens={tokens}/{budget}");
    }
    Bundle {
        text,
        tokens,
        facts,
    }
}

/// `ms` since the epoch as an RFC 3339 UTC second.
pub fn rfc3339_ms(ms: u64) -> String {
    let secs = (ms / 1000) as i64;
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

/// An RFC 3339 `YYYY-MM-DDTHH:MM:SS[.frac]Z` as seconds since the epoch.
pub fn parse_rfc3339_secs(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' {
        return None;
    }
    let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (hh, mm, ss) = (n(11..13)?, n(14..16)?, n(17..19)?);
    if !s.ends_with('Z') || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // days_from_civil.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    u64::try_from(days * 86_400 + hh * 3600 + mm * 60 + ss).ok()
}

#[cfg(test)]
mod tests;
