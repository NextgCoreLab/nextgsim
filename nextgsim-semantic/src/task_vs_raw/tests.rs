use std::sync::OnceLock;

use rand::seq::SliceRandom;

use super::*;

/// The seed-444 `all` run, computed once and shared read-only.
fn seed_444() -> &'static (Sweep, Vec<Row>) {
    static RUN: OnceLock<(Sweep, Vec<Row>)> = OnceLock::new();
    RUN.get_or_init(|| {
        let sweep = Sweep::new(DEFAULT_NOISE_SEED);
        let rows = sweep.run(&Arm::ALL);
        (sweep, rows)
    })
}

fn rows_of(table: &str) -> Vec<&str> {
    table.lines().filter(|l| !l.starts_with('#')).collect()
}

/// Prints the table; the golden below is this output.
#[test]
#[ignore = "probe: prints the seed-444 table"]
fn print_the_table() {
    let (_, rows) = seed_444();
    print!("{}", render(Mode::All, DEFAULT_NOISE_SEED, rows));
}

/// Acceptance: two runs with the same seed print byte-identical tables. Fails
/// if the channel draws from an unseeded RNG (revert arm 479-thread-rng).
#[test]
fn two_runs_with_the_same_seed_print_byte_identical_tables() {
    let (_, first) = seed_444();
    let second = Sweep::new(DEFAULT_NOISE_SEED).run(&Arm::ALL);
    assert_eq!(
        render(Mode::All, DEFAULT_NOISE_SEED, first),
        render(Mode::All, DEFAULT_NOISE_SEED, &second)
    );
}

/// The committed seed-444 table, row by row.
#[test]
fn the_seed_444_table_matches_the_committed_golden_row_by_row() {
    let (_, rows) = seed_444();
    let table = render(Mode::All, DEFAULT_NOISE_SEED, rows);
    let golden = include_str!("golden_seed444.txt");
    let (got, want) = (
        table.lines().collect::<Vec<_>>(),
        golden.lines().collect::<Vec<_>>(),
    );
    assert_eq!(got.len(), want.len(), "line count\n{table}");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "line {}", i + 1);
    }
    // 4 keeps x 5 SNRs per arm, 3 arms, then a delta header and as many
    // delta lines.
    assert_eq!(rows.len(), Arm::ALL.len() * KEEPS.len() * SNRS_DB.len());
    assert_eq!(
        table.lines().filter(|l| l.starts_with("delta ")).count(),
        KEEPS.len() * SNRS_DB.len() + 1
    );
}

/// A different seed changes the noise draws, and so the table; the generator
/// (and so every noiseless fit) is the same.
#[test]
fn a_different_seed_changes_the_noise_draws() {
    let draw =
        |seed| AwgnChannel::new(0.0).transmit_with_rng(&[1.0; 16], &mut noise_rng(seed, 16, 0));
    assert_eq!(draw(444), draw(444));
    assert_ne!(draw(444), draw(445));
    let (sweep, rows) = seed_444();
    let other = Sweep::new(445);
    let other_rows = other.run(&[Arm::TaskEncoder]);
    assert_ne!(
        rows_of(&render(Mode::Arm(Arm::TaskEncoder), 0, &other_rows)),
        rows_of(&render(
            Mode::Arm(Arm::TaskEncoder),
            0,
            &rows[..other_rows.len()]
        ))
    );
    assert_eq!(
        sweep.classifier(Arm::TaskEncoder, 16).centroids(),
        other.classifier(Arm::TaskEncoder, 16).centroids()
    );
}

/// Acceptance: per arm and keep, 20 dB scores at least 0 dB, and no step up
/// in SNR loses more than 0.01.
#[test]
fn accuracy_does_not_fall_as_snr_rises() {
    let (_, rows) = seed_444();
    for arm in Arm::ALL {
        for &keep in KEEPS {
            let curve: Vec<f64> = rows
                .iter()
                .filter(|r| r.arm == arm && r.keep == keep)
                .map(|r| r.score.top1())
                .collect();
            assert_eq!(curve.len(), SNRS_DB.len());
            assert!(
                curve[curve.len() - 1] >= curve[0],
                "{} keep {keep}: {curve:?}",
                arm.name()
            );
            for w in curve.windows(2) {
                assert!(w[1] >= w[0] - 0.01, "{} keep {keep}: {curve:?}", arm.name());
            }
        }
    }
}

/// Acceptance (baseline arm): at every (keep, SNR) all three arms sent the
/// same bytes, 2 per kept value, measured from the encoded vectors; the deltas
/// are reported in the table.
#[test]
fn every_arm_sends_the_same_bytes_at_each_point() {
    let (_, rows) = seed_444();
    for &keep in KEEPS {
        for &snr in SNRS_DB {
            let at: Vec<&Row> = rows
                .iter()
                .filter(|r| r.keep == keep && r.snr_db == snr)
                .collect();
            assert_eq!(at.len(), Arm::ALL.len());
            assert!(
                at.iter().all(|r| r.bytes == keep * BYTES_PER_FEATURE),
                "{at:?}"
            );
            assert!(at.iter().all(|r| r.score.samples() == HELD_OUT));
        }
    }
    let (sweep, _) = seed_444();
    for arm in Arm::ALL {
        assert!(sweep.held_out(arm, 16).iter().all(|(_, s)| s.len() == 16));
    }
}

/// Acceptance: the classifier scores 1.0 on its own noiseless centroids.
#[test]
fn nearest_centroid_scores_its_own_centroids_perfectly() {
    let (sweep, _) = seed_444();
    for arm in Arm::ALL {
        for &keep in KEEPS {
            let classifier = sweep.classifier(arm, keep);
            let centroids = classifier.centroids().to_vec();
            assert_eq!(centroids.len(), CLASSES);
            let s = score(
                &classifier,
                centroids
                    .into_iter()
                    .enumerate()
                    .map(|(c, v)| (c as u8, v.clone(), v)),
            );
            assert_eq!(s.samples(), CLASSES as u64);
            assert_eq!(s.top1(), 1.0, "{} keep {keep}", arm.name());
        }
    }
}

/// Acceptance: fitted on shuffled labels, the classifier scores at chance
/// (1/8 within 0.04, about five binomial sigmas at 2000 samples). Fails if the
/// score is computed with `top_k_accuracy` (revert arm 479-top-k-scorer), which
/// compares feature indices and never looks at the label.
#[test]
fn shuffled_labels_score_at_chance() {
    let (sweep, _) = seed_444();
    let arm = Arm::TaskEncoder;
    let keep = 32;
    let encoder = sweep.model.encoder(arm.encoder());
    let train: Vec<(u8, Vec<f32>)> = sweep
        .model
        .generator()
        .training_set()
        .map(|(l, x)| (l, encoder.encode(&x, keep)))
        .collect();
    let mut labels: Vec<u8> = train.iter().map(|(l, _)| *l).collect();
    labels.shuffle(&mut ChaCha8Rng::seed_from_u64(479));
    let shuffled = NearestCentroid::fit(labels.into_iter().zip(train.into_iter().map(|(_, x)| x)));
    let held_out = sweep.held_out(arm, keep);
    let s = score(
        &shuffled,
        held_out.into_iter().map(|(l, x)| (l, x.clone(), x)),
    );
    let chance = 1.0 / CLASSES as f64;
    assert!(
        (s.top1() - chance).abs() < 0.04,
        "top1 {} vs chance {chance}",
        s.top1()
    );
    // The same arm with the true labels is far above chance.
    let true_fit = sweep.classifier(arm, keep);
    let held_out = sweep.held_out(arm, keep);
    let s = score(
        &true_fit,
        held_out.into_iter().map(|(l, x)| (l, x.clone(), x)),
    );
    assert!(s.top1() > 0.5, "{}", s.top1());
}

/// Acceptance (modes): each single-arm mode's rows are byte-identical to the
/// matching rows of `all`.
#[test]
fn each_single_arm_mode_prints_exactly_its_rows_of_all() {
    let (_, all_rows) = seed_444();
    let all = render(Mode::All, DEFAULT_NOISE_SEED, all_rows);
    let sweep = Sweep::new(DEFAULT_NOISE_SEED);
    for arm in Arm::ALL {
        let table = render(Mode::Arm(arm), DEFAULT_NOISE_SEED, &sweep.run(&[arm]));
        let own: Vec<&str> = rows_of(&table).into_iter().skip(1).collect();
        let want: Vec<&str> = all
            .lines()
            .filter(|l| l.starts_with(&format!("{} ", arm.name())))
            .collect();
        assert_eq!(own.len(), KEEPS.len() * SNRS_DB.len());
        assert_eq!(own, want, "{}", arm.name());
        assert!(
            !table.contains("delta"),
            "a single-arm mode prints no deltas"
        );
    }
}

const ON: &str = "demo:\n  enabled: true\n  d444c-task-vs-raw:\n    enabled: true\n    mode: all\n";

#[test]
fn an_enabled_demo_selects_its_mode() {
    let (runtime, mode) = prepare(ON).unwrap();
    assert_eq!(mode, Some(Mode::All));
    assert_eq!(
        runtime.snapshot().banner().as_deref(),
        Some("DEMO MODE ACTIVE: d444c-task-vs-raw=all")
    );
    for m in MODES {
        let (_, mode) = prepare(&ON.replace("mode: all", &format!("mode: {m}"))).unwrap();
        assert_eq!(mode.map(Mode::name), Some(*m));
    }
}

/// Acceptance (kill switch): `demo.enabled: false` overrides the demo's own
/// `enabled: true`: no mode, so no table, and no banner.
#[test]
fn the_kill_switch_overrides_an_enabled_demo() {
    for text in [
        ON.replace("demo:\n  enabled: true", "demo:\n  enabled: false"),
        ON.replace(
            "    enabled: true\n    mode",
            "    enabled: false\n    mode",
        ),
        "demo: {}\n".to_string(),
        "{}\n".to_string(),
    ] {
        let (runtime, mode) = prepare(&text).unwrap();
        assert_eq!(mode, None, "{text}");
        assert_eq!(runtime.snapshot().banner(), None, "{text}");
        let mut metrics = String::new();
        runtime.render_metrics(&mut metrics);
        assert!(metrics.is_empty(), "{text}");
    }
}

/// Acceptance (invalid mode): `mode: bogus` is refused by name, even with the
/// kill switch off; so are an unknown key and an unknown root field.
#[test]
fn an_unknown_mode_is_refused_by_name() {
    for text in [
        ON.replace("mode: all", "mode: bogus"),
        ON.replace("mode: all", "mode: bogus")
            .replace("demo:\n  enabled: true", "demo:\n  enabled: false"),
    ] {
        match prepare(&text) {
            Err(ConfigError::Demo(DemoConfigError::UnknownMode { demo, mode, .. })) => {
                assert_eq!((demo.as_str(), mode.as_str()), (KEY, "bogus"));
            }
            other => panic!("{text}: {other:?}"),
        }
    }
    assert!(matches!(
        prepare(&ON.replace("d444c-task-vs-raw", "d444c-typo")),
        Err(ConfigError::Demo(DemoConfigError::UnknownKey { .. }))
    ));
    assert!(matches!(
        prepare(&format!("seed: 1\n{ON}")),
        Err(ConfigError::UnknownField(_))
    ));
}
