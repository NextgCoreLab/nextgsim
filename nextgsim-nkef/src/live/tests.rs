use super::*;

const T0: u64 = 1_794_000_000_000;
const NCI: u64 = 0x10;
const GUAMI: &str = "00101-02-155-2a";

/// The facts of one fresh run: RLS, NGAP, amfd and the NRF all reported.
fn full_run() -> LiveStore {
    let mut s = LiveStore::new(T0);
    s.set_serving_nci(NCI);
    s.upsert(
        Fact::new(
            Domain::Ran,
            FactKind::CellLoad,
            "gnb-rls",
            "cell-16",
            T0 + 1_000,
        )
        .with("connected_ues", 1)
        .with("throughput_mbps", "0.001"),
    );
    s.upsert(
        Fact::new(
            Domain::Ran,
            FactKind::UeRadio,
            "gnb-rls",
            "ue-1",
            T0 + 1_000,
        )
        .with("cell", 16)
        .with("rsrp_dbm", -80),
    );
    s.upsert(Fact::new(Domain::N2, FactKind::Gnb, "gnb-ngap", "gnb-1", T0 + 500).with("nci", NCI));
    s.upsert(
        Fact::new(Domain::N2, FactKind::Amf, "gnb-ngap", "amf-0", T0 + 600)
            .with("amf_name", "nextgcore-amf")
            .with("guami", GUAMI),
    );
    s.upsert(
        Fact::new(Domain::N2, FactKind::UeN2, "gnb-ngap", "ue-1", T0 + 900)
            .with("amf", "amf-0")
            .with("amf_ue_ngap_id", 1),
    );
    s.upsert(
        Fact::new(
            Domain::Sba,
            FactKind::UeLocation,
            "amf",
            "imsi-001010000000001",
            T0 + 2_000,
        )
        .with("nrCellId", "000000010")
        .with("ueLocationTimestamp", "2026-11-11T08:00:01Z")
        .with("ueLocationTimestampEpoch", T0 / 1000 + 1),
    );
    s.upsert(
        Fact::new(
            Domain::Sba,
            FactKind::UeConnectivity,
            "amf",
            "imsi-001010000000001",
            T0 + 2_000,
        )
        .with("cmState", "CONNECTED"),
    );
    s.upsert(
        Fact::new(
            Domain::Sba,
            FactKind::NfInstance,
            "nrf",
            "nf-amf",
            T0 + 3_000,
        )
        .with("nfType", "AMF")
        .with("nfStatus", "REGISTERED")
        .with("nfInstanceId", "3fa85f64-5717-4562-b3fc-2c963f66afa6")
        .with("guamis", format!("00101-01-001-01,{GUAMI}")),
    );
    s.upsert(
        Fact::new(
            Domain::Sba,
            FactKind::NfInstance,
            "nrf",
            "nf-bsf",
            T0 + 3_000,
        )
        .with("nfType", "BSF")
        .with("nfStatus", "REGISTERED")
        .with("nfInstanceId", "0b0e8a5e-0000-4000-8000-00000000b5f0"),
    );
    s
}

const Q: &str = "is my route congested and which core functions serve me?";

#[test]
fn a_full_run_answers_with_every_domain_joined_and_consistent() {
    let b = full_run().bundle(Q, Consumer::Internal, 400, T0 + 5_000);
    let header = b.text.lines().next().unwrap();
    assert!(
        header.starts_with(
            "NKEF bundle consumer=internal domains=ran,n2,sba consistent=true \
             path=ue>cell>gnb>amf>nf-instance cm=CONNECTED stale=false \
             amf_nf_instance=3fa85f64-5717-4562-b3fc-2c963f66afa6 tokens="
        ),
        "{header}"
    );
    assert_eq!(b.facts, 9, "{}", b.text);
    assert!(
        b.text.contains("producer=nrf"),
        "every fact names its producer"
    );
    assert!(b.text.contains("age=2s"), "and its age: {}", b.text);
}

#[test]
fn the_path_stops_where_the_join_does() {
    let mut s = full_run();
    // The NRF instance names another GUAMI: the AMF hop holds, the last does not.
    s.upsert(
        Fact::new(
            Domain::Sba,
            FactKind::NfInstance,
            "nrf",
            "nf-amf",
            T0 + 3_000,
        )
        .with("nfType", "AMF")
        .with("nfInstanceId", "x")
        .with("guamis", "00101-09-009-09"),
    );
    assert_eq!(s.path(), (vec!["ue", "cell", "gnb", "amf"], None));
}

#[test]
fn an_nf_the_nrf_stops_returning_expires_and_is_kept() {
    let mut s = full_run();
    let present: BTreeSet<String> = ["nf-amf".to_string()].into();
    let gone = s.expire_missing(FactKind::NfInstance, "nrf", &present, T0 + 20_000);
    assert_eq!(gone, vec!["nf-bsf".to_string()]);
    let bsf = s.get(FactKind::NfInstance, "nf-bsf").unwrap();
    assert_eq!(bsf.expired_at_ms, Some(T0 + 20_000));
    let b = s.bundle("bsf", Consumer::Internal, 400, T0 + 21_000);
    assert!(
        b.text.contains("subject=nf-bsf") && b.text.contains("expired_at=2026-11-06T"),
        "{}",
        b.text
    );
    // Before the stop it had none.
    let b = full_run().bundle("bsf", Consumer::Internal, 400, T0 + 21_000);
    assert!(!b.text.contains("expired_at"), "{}", b.text);
    // A second poll keeps the first expiry time.
    assert!(s
        .expire_missing(FactKind::NfInstance, "nrf", &present, T0 + 25_000)
        .is_empty());
    assert_eq!(
        s.get(FactKind::NfInstance, "nf-bsf").unwrap().expired_at_ms,
        Some(T0 + 20_000)
    );
}

#[test]
fn without_a_timestamped_amf_location_consistency_is_unknown() {
    let mut s = full_run();
    let mut loc = s
        .get(FactKind::UeLocation, "imsi-001010000000001")
        .unwrap()
        .clone();
    loc.attrs.remove("ueLocationTimestamp");
    loc.attrs.remove("ueLocationTimestampEpoch");
    s.upsert(loc.clone());
    assert_eq!(s.consistent(), "unknown");
    assert_eq!(s.stale(), "unknown");
    // A different cell is inconsistent.
    s.upsert(
        loc.with("nrCellId", "000000020")
            .with("ueLocationTimestamp", "x")
            .with("ueLocationTimestampEpoch", T0 / 1000),
    );
    assert_eq!(s.consistent(), "false");
}

#[test]
fn a_location_from_before_this_run_is_stale() {
    let mut s = full_run();
    let loc = s
        .get(FactKind::UeLocation, "imsi-001010000000001")
        .unwrap()
        .clone()
        .with("ueLocationTimestampEpoch", T0 / 1000 - 600);
    s.upsert(loc);
    assert_eq!(s.stale(), "true");
}

#[test]
fn a_baseline_store_has_no_live_domain() {
    let b = LiveStore::new(T0).bundle(Q, Consumer::Internal, 400, T0);
    assert!(
        b.text.starts_with(
            "NKEF bundle consumer=internal domains=none consistent=unknown path=none \
             cm=unknown stale=unknown"
        ),
        "{}",
        b.text
    );
    assert_eq!(b.facts, 0);
}

#[test]
fn an_external_consumer_gets_area_congestion_and_nothing_identifying() {
    let mut s = full_run();
    let b = s.bundle(Q, Consumer::External, 400, T0 + 5_000);
    assert_eq!(
        b.text,
        "NKEF bundle consumer=external areas=1 tokens=8/400\narea-1 congestion=Low age=4s"
    );
    // A ping burst raises the level.
    s.upsert(
        Fact::new(
            Domain::Ran,
            FactKind::CellLoad,
            "gnb-rls",
            "cell-16",
            T0 + 6_000,
        )
        .with("throughput_mbps", "1.7"),
    );
    let b = s.bundle(Q, Consumer::External, 400, T0 + 6_000);
    assert!(b.text.contains("area-1 congestion=Medium"), "{}", b.text);
    for leak in [
        "imsi-", "ncgi", "nrCellId", "tac", "nci", "ngap_id", "172.23.", "3fa85f64", "cell-",
        "amf", "gnb",
    ] {
        assert!(!b.text.contains(leak), "{leak} leaked: {}", b.text);
    }
}

#[test]
fn the_budget_is_a_hard_cap_first_fact_included() {
    let s = full_run();
    for budget in [0usize, 5, 30, 31, 40, 60, 400] {
        let b = s.bundle(Q, Consumer::Internal, budget, T0 + 5_000);
        assert!(
            b.tokens <= budget,
            "{budget}: {} > budget\n{}",
            b.tokens,
            b.text
        );
        assert_eq!(token_cost(&b.text), b.tokens, "{budget}: the count is true");
    }
    // A budget that holds the header but not one fact carries no fact.
    let header_only = token_cost(
        s.bundle(Q, Consumer::Internal, 400, T0)
            .text
            .lines()
            .next()
            .unwrap(),
    );
    let b = s.bundle(Q, Consumer::Internal, header_only, T0);
    assert_eq!(b.facts, 0, "{}", b.text);
}

#[test]
fn ranking_puts_the_facts_the_question_names_first() {
    let b = full_run().bundle("bsf instance", Consumer::Internal, 400, T0 + 5_000);
    let first = b.text.lines().nth(1).unwrap();
    assert!(first.contains("subject=nf-bsf"), "{}", b.text);
}

#[test]
fn rfc3339_round_trips() {
    assert_eq!(rfc3339_ms(T0), "2026-11-06T21:20:00Z");
    assert_eq!(parse_rfc3339_secs("2026-11-06T21:20:00Z"), Some(T0 / 1000));
    assert_eq!(parse_rfc3339_secs("1970-01-01T00:00:00Z"), Some(0));
    assert_eq!(
        parse_rfc3339_secs("2024-02-29T12:34:56Z"),
        Some(1_709_210_096)
    );
    assert_eq!(parse_rfc3339_secs("not a time"), None);
}

#[test]
fn congestion_levels_follow_the_project_rule() {
    assert_eq!(congestion_level(0.0), "Low");
    assert_eq!(congestion_level(LOW_BELOW_MBPS), "Medium");
    assert_eq!(congestion_level(MEDIUM_BELOW_MBPS), "High");
}
