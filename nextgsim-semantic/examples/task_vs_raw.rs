//! Demo D444C (nextgsim#479): the seeded accuracy-vs-bytes-vs-SNR table of the
//! task encoder vs raw at equal bytes, under an **emulated** AWGN channel.
//! Crate-level and non-normative; see `nextgsim_semantic::task_vs_raw`.
//!
//! ```text
//! cargo run -p nextgsim-semantic --features demo-d444c-task-vs-raw \
//!     --example task_vs_raw -- --config config/demo-task-vs-raw/task_vs_raw.yaml --seed 444
//! ```
//!
//! The config holds only the `demo:` block (nextgsim#458). The table goes to
//! stdout; the `WARN DEMO MODE ACTIVE` banner, the one-shot
//! `nextg_demo_enabled` series and the `[demo:d444c-task-vs-raw]` log lines go
//! to stderr (the process has no `/metrics` and opens no listener). With the
//! demo off (no block, the kill switch off, or the key off) it prints no table
//! and exits 0; a refused config exits 2.

use std::process::ExitCode;

use nextgsim_demo::demo_log;
use nextgsim_semantic::task_vs_raw::{prepare, render, Sweep, DEFAULT_NOISE_SEED, KEY};

struct Args {
    config: String,
    seed: u64,
}

fn args() -> Result<Args, String> {
    let usage = "usage: task_vs_raw --config <file> [--seed <u64>]";
    let mut config = None;
    let mut seed = DEFAULT_NOISE_SEED;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--config" | "-c" => config = Some(it.next().ok_or(usage)?),
            "--seed" => {
                let v = it.next().ok_or(usage)?;
                seed = v.parse().map_err(|_| format!("--seed {v:?}: not a u64"))?;
            }
            _ => return Err(format!("unknown argument {a:?}; {usage}")),
        }
    }
    Ok(Args {
        config: config.ok_or(usage)?,
        seed,
    })
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = match args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("task_vs_raw: {e}");
            return ExitCode::from(2);
        }
    };
    let text = match std::fs::read_to_string(&args.config) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("task_vs_raw: {}: {e}", args.config);
            return ExitCode::from(2);
        }
    };
    let (runtime, mode) = match prepare(&text) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("task_vs_raw: {}: {e}\n{e:?}", args.config);
            return ExitCode::from(2);
        }
    };
    let Some(mode) = mode else {
        eprintln!(
            "task_vs_raw: demo {KEY} is not enabled (demo.enabled and demo.{KEY}.enabled): no table"
        );
        return ExitCode::SUCCESS;
    };
    let mut metrics = String::new();
    runtime.render_metrics(&mut metrics);
    if nextgsim_demo::install(runtime).is_err() {
        eprintln!("task_vs_raw: demo state already installed");
        return ExitCode::from(2);
    }
    for line in metrics.lines().filter(|l| !l.starts_with('#')) {
        demo_log!(info, KEY, "{line} (one-shot: this process has no /metrics)");
    }
    demo_log!(
        info,
        KEY,
        "sweep: mode {}, noise seed {}, emulated AWGN (non-normative)",
        mode.name(),
        args.seed
    );
    let rows = Sweep::new(args.seed).run(&mode.arms());
    print!("{}", render(mode, args.seed, &rows));
    demo_log!(info, KEY, "printed {} rows", rows.len());
    ExitCode::SUCCESS
}
