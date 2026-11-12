use super::*;

#[test]
#[ignore = "calibration probe: prints the accuracy table"]
fn calibration_probe() {
    let mut m = D444aModel::new(DEFAULT_SEED);
    println!("eig head {:?}", &m.pca().eigenvalues()[..6]);
    for keep in [8, 16, 24, 32, 40, 64, 128] {
        let p = m.accuracy(EncoderId::Pca, keep, 0..2000);
        let t = m.accuracy(EncoderId::TopVarianceRaw, keep, 0..2000);
        let mp = m.accuracy(EncoderId::MeanPool, keep, 0..2000);
        println!("keep={keep} pca={p:.3} topvar={t:.3} meanpool={mp:.3}");
    }
}

/// The link budget is the issue's arithmetic: 64 kbps UL at 50 Hz, 80 %
/// headroom, gives 51 200 bit/s, 128 B per datagram, 128 - 28 - 16 = 84 B of
/// features, 42 features, rounded down to the step: 40. The establishment's
/// 2 Mbps leaves the full 128.
#[test]
fn keep_follows_the_uplink_session_ambr() {
    assert_eq!(rate_target_bps(64_000), 51_200);
    assert_eq!(keep_for_link_budget(64_000, 50), 40);
    assert_eq!(datagram_ip_bytes(40), 124);
    assert!(datagram_ip_bytes(40) * 8 * 50 <= 51_200);
    assert_eq!(keep_for_link_budget(2_000_000, 50), FEATURE_DIM);
    // Below the floor the application still sends MIN_KEEP.
    assert_eq!(keep_for_link_budget(1_000, 50), MIN_KEEP);
    // The raw control at 128 features needs 120 kbit/s: above a 64 kbps QER.
    assert_eq!(datagram_ip_bytes(FEATURE_DIM) * 8 * 50, 120_000);
}

/// The header bytes are the documented layout, and decode inverts encode up to
/// the i16 quantisation.
#[test]
fn a_datagram_has_the_documented_layout_and_round_trips() {
    let d = FeatureDatagram {
        encoder: EncoderId::Pca,
        seq: 0x0102_0304,
        label: 5,
        values: vec![1.0, -0.5, 0.25],
    };
    let bytes = d.encode();
    assert_eq!(bytes.len(), HEADER_LEN + 3 * BYTES_PER_FEATURE);
    assert_eq!(&bytes[..12], &[b'D', b'4', 1, 1, 1, 2, 3, 4, 5, 3, 0, 0]);
    // value 1.0 is the largest, so it is i16::MAX.
    assert_eq!(&bytes[16..18], &i16::MAX.to_be_bytes());
    let back = FeatureDatagram::decode(&bytes).unwrap();
    assert_eq!(
        (back.encoder, back.seq, back.label),
        (EncoderId::Pca, 0x0102_0304, 5)
    );
    for (a, b) in back.values.iter().zip(&d.values) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
    assert_eq!(
        FeatureDatagram::decode(&bytes[..10]),
        Err(WireError::Truncated(10))
    );
    assert_eq!(
        FeatureDatagram::decode(&bytes[..17]),
        Err(WireError::Truncated(17))
    );
    let mut bad = bytes.clone();
    bad[0] = b'X';
    assert_eq!(FeatureDatagram::decode(&bad), Err(WireError::BadMagic));
    bad = bytes;
    bad[3] = 9;
    assert_eq!(
        FeatureDatagram::decode(&bad),
        Err(WireError::UnknownEncoder(9))
    );
}

/// The UE and the sink build their models independently: the same seed must
/// give the same samples, drawn in any order.
#[test]
fn the_generator_is_deterministic_and_random_access() {
    let a = FeatureGenerator::new(DEFAULT_SEED);
    let b = FeatureGenerator::new(DEFAULT_SEED);
    let late = b.sample(777);
    assert_eq!(a.sample(3), b.sample(3));
    assert_eq!(a.sample(777), late);
    assert_ne!(a.sample(3).1, a.sample(4).1);
    assert_ne!(FeatureGenerator::new(1).sample(3).1, a.sample(3).1);
    let (label, x) = a.sample(9);
    assert!(usize::from(label) < CLASSES);
    assert_eq!(x.len(), FEATURE_DIM);
}

/// The pinned generator's ground truth at keep 40: the PCA reaches the floor
/// and beats both equal-bytes baselines. The margins come from the generator's
/// spectrum (see the module docs), not from any learned semantics.
#[test]
fn at_keep_40_pca_reaches_the_floor_and_beats_both_baselines() {
    let mut m = D444aModel::new(DEFAULT_SEED);
    let ev = m.pca().eigenvalues();
    assert!(ev.windows(2).all(|w| w[0] >= w[1]), "eigenvalues sorted");
    let pca = m.accuracy(EncoderId::Pca, 40, 0..600);
    let top = m.accuracy(EncoderId::TopVarianceRaw, 40, 0..600);
    let pool = m.accuracy(EncoderId::MeanPool, 40, 0..600);
    assert!(pca >= PINNED_ACCURACY_FLOOR_KEEP40, "pca {pca}");
    assert!(pca > top && pca > pool, "pca {pca} top {top} pool {pool}");
}

fn datagram(m: &D444aModel, id: EncoderId, seq: u32, keep: usize) -> FeatureDatagram {
    let (label, x) = m.generator().sample(u64::from(seq));
    FeatureDatagram {
        encoder: id,
        seq,
        label,
        values: m.encoder(id).encode(&x, keep),
    }
}

/// The sink's rows: epoch 0 complete at keep 128, epoch 1 at keep 40 with
/// every third datagram lost, closed by epoch 2's first datagram. The keep-40
/// row carries the equal-bytes baselines; byte and loss counts are exact.
#[test]
fn the_scorer_counts_bytes_losses_and_baselines_per_epoch() {
    let model = D444aModel::new(DEFAULT_SEED);
    let probe = D444aModel::new(DEFAULT_SEED);
    let mut s = EpochScorer::new(model, 60, 50);
    let mut rows = Vec::new();
    for seq in 0..60u32 {
        let d = datagram(&probe, EncoderId::Pca, seq, 128);
        let wire = d.encode();
        let d = FeatureDatagram::decode(&wire).unwrap();
        rows.extend(s.on_datagram(&d, wire.len(), "semantic"));
    }
    assert!(rows.is_empty(), "epoch 0 is still open");
    for seq in (60..120u32).filter(|s| s % 3 != 0) {
        let d = datagram(&probe, EncoderId::Pca, seq, 40);
        let wire = d.encode();
        let d = FeatureDatagram::decode(&wire).unwrap();
        rows.extend(s.on_datagram(&d, wire.len(), "semantic"));
    }
    let d = datagram(&probe, EncoderId::Pca, 120, 40);
    let wire = d.encode();
    rows.extend(s.on_datagram(
        &FeatureDatagram::decode(&wire).unwrap(),
        wire.len(),
        "semantic",
    ));
    assert_eq!(rows.len(), 2);
    let (a, b) = (&rows[0], &rows[1]);
    assert_eq!((a.epoch, a.keep_max, a.received, a.lost), (0, 128, 60, 0));
    assert_eq!(a.ip_bytes, 60 * datagram_ip_bytes(128) as u64);
    assert_eq!(
        (b.epoch, b.keep_min, b.keep_max, b.received, b.lost),
        (1, 40, 40, 40, 20)
    );
    assert_eq!(b.ip_bytes, 40 * 124);
    // 4 960 B over 60 datagrams at 50 Hz = 1.2 s: 33.07 kbit/s.
    assert!((b.ul_kbps - 33.067).abs() < 0.01, "{}", b.ul_kbps);
    assert!(b.top_variance_raw_accuracy.is_some() && b.mean_pool_accuracy.is_some());
    let line = b.to_string();
    assert!(
        line.starts_with(
            "epoch=1 mode=semantic encoder=pca keep=40 received=40 lost=20 ip_bytes=4960"
        ),
        "{line}"
    );
    assert!(line.contains("baseline top-variance-raw acc="), "{line}");
}

/// A raw-mode epoch scores the raw encoder and carries no baseline columns;
/// a restarted sender (sequence back to 0) starts a fresh span.
#[test]
fn a_raw_epoch_has_no_baselines_and_a_restart_resets_the_span() {
    let probe = D444aModel::new(DEFAULT_SEED);
    let mut s = EpochScorer::new(D444aModel::new(DEFAULT_SEED), 10, 50);
    for seq in 200..210u32 {
        let d = datagram(&probe, EncoderId::Raw, seq, 128);
        assert!(s.on_datagram(&d, d.encode().len(), "raw").is_none());
    }
    let d = datagram(&probe, EncoderId::Raw, 0, 128);
    let row = s
        .on_datagram(&d, d.encode().len(), "raw")
        .expect("restart closes epoch 20");
    assert_eq!((row.epoch, row.received, row.lost), (20, 10, 0));
    assert_eq!(row.top_variance_raw_accuracy, None);
    assert!(!row.to_string().contains("baseline"));
    let row = s.close("raw").unwrap();
    assert_eq!((row.epoch, row.received, row.lost), (0, 1, 9));
}
