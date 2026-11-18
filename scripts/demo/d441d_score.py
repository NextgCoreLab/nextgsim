#!/usr/bin/env python3
"""Score the episodes of demo D441D, predictive N2 handover vs A3 (nextgsim#467).

One seeded episode is one UE driving the committed route from gnb-west toward
gnb-east. This scores it from three captured logs (gnb-west, gnb-east, amfd).
Each log line starts with a timestamp, either `<RFC 3339> <text>`
(`kubectl logs --timestamps`) or `<service> | <RFC 3339> <text>`
(`docker compose logs -t`). ANSI colour codes are ignored. The strings matched
are the ones the code emits (nextgsim-gnb nwdaf/task.rs, rrc/task.rs,
ngap/task.rs; nextgcore amfd).

    d441d_score.py score --arm lsq --seed 3 --west W --east E --amf A [--ue U]
        One TSV row on stdout (the columns are COLUMNS below). The `chain`
        column is `ok` or the first missing link, so a failed seed names its
        reason. With --ue, the episode's window (--window-ms, 120 s) starts at
        the UE's `Simulated motion started` line: a HandoverNotify after it
        does not count (decision Q4).
    d441d_score.py table --arm lsq --seeds 1-24 --min-chains 22 ROWS.tsv
        The per-arm summary (median and p10 of the lead's bounds, ping-pongs,
        returns, failures, fallbacks, stranded, exhausted) as Markdown. Exit 1
        when fewer than --min-chains seeds held the chain, when a planned seed
        has no row, when a seed has two rows or a row is not planned, or when
        a row is malformed.

Outcomes, per arm:
  lsq, onnx  predictive (the predictor's chain held), fallback (A3 moved the
             UE after the grace), stranded, predictor-mislabelled (for onnx:
             `predictor=fallback`, no verified model served it), failed (no
             HandoverNotify in the window, or a handover failure; exhausted
             attempts end here too), broken (a link of the chain is missing
             although the UE was handed over).
  a3         a3 (the UE's A3 report triggered it, the replica's range holds
             the report's time), a3-outside (the range does not), predicted
             (a predictor line in the A3 arm), failed, broken.
  off        a3 (the base A3 handover, and no demo line at all), demo-line,
             failed, broken.

A ping-pong is the UE handed back to gnb-west (gnb-west sends a HANDOVER
NOTIFY as the target) within --pingpong-ms of the first HandoverNotify at
amfd; `returns` counts every hand-back in the episode.

Standard library only; run it with `python3 -I`.
"""
import argparse
import datetime as dt
import re
import statistics
import sys

COLUMNS = (
    "seed", "arm", "outcome", "chain", "predictor", "lead_lo_ms", "lead_hi_ms",
    "a3_lo_ms", "a3_hi_ms", "a3_report_ms", "pingpong", "returns", "failures",
    "fallbacks", "stranded", "exhausted",
)
DEMO = "[demo:d441d-predictive-ho]"
TARGET_CELL = "32"  # gnb-east: nci 0x20, the cell the predictor and A3 name
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(
    r"^(?:\S+\s+\|\s+)?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:?\d\d)\s?(.*)$"
)

RE_PREDICTION = re.compile(
    re.escape(DEMO) + r" NWDAF: handover prediction ue=\S+ predictor=(\w+) target=" + TARGET_CELL + r" ")
RE_RECOMMENDED = re.compile(r"RRC: NWDAF-recommended handover for UE \S+ to cell " + TARGET_CELL + r" ")
RE_A3_REPORT = re.compile(r"measurement-triggered N2 handover to PCI " + TARGET_CELL + r":")
RE_REQUIRED = re.compile(r"Sent Handover Required: amf_ue_ngap_id=")
RE_AMF_NOTIFY = re.compile(r"HandoverNotify from association [0-9]+ \(amf_ue_ngap_id=")
RE_SENT_NOTIFY = re.compile(r"Sent Handover Notify: amf_ue_ngap_id=")
RE_A3_WOULD = re.compile(
    re.escape(DEMO) + r" A3 would trigger t=(\d+)\.\.(\d+) ue=\S+ target=" + TARGET_CELL + r" ")
RE_LEAD = re.compile(
    re.escape(DEMO) + r" lead time ue=\S+ target=" + TARGET_CELL
    + r" predictive_t=\d+ a3_t=\d+\.\.\d+ lead_ms=(-?\d+)\.\.(-?\d+)")
RE_FALLBACK = re.compile(re.escape(DEMO) + r" A3 fallback=a3 n=")
RE_STRANDED = re.compile(re.escape(DEMO) + r" stranded ue=")
RE_EXHAUSTED = re.compile(re.escape(DEMO) + r" NWDAF: handover prediction exhausted ue=")
RE_ANY_DEMO = re.compile(r"DEMO MODE ACTIVE|\[demo:")
# A handover that did not complete: preparation refused or cancelled at the
# source, a TNGRELOC timer, the UE back at a cell by re-establishment, or amfd
# relaying a HandoverFailure.
RE_GNB_FAILURE = re.compile(
    r"Handover Preparation Failure: |Sent Handover Cancel: |TNGRELOC(prep|overall) expired for UE"
    r"|RRCReestablishmentComplete verified")
RE_AMF_FAILURE = re.compile(r"HandoverFailure from association [0-9]+ |HandoverNotify from association [0-9]+ rejected")
RE_ROUTE_START = re.compile(r"Simulated motion started")


def parse_ms(stamp, frac, zone):
    """RFC 3339 to Unix milliseconds (any fraction length, Z or an offset)."""
    zone = "+00:00" if zone == "Z" else (zone if ":" in zone else zone[:3] + ":" + zone[3:])
    t = dt.datetime.fromisoformat(stamp + zone)
    ms = int(t.timestamp()) * 1000
    if frac:
        ms += int((frac + "000")[:3])
    return ms


def read_log(path):
    """[(ms, text)] for every line that carries a timestamp, in file order."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = STAMP.match(ANSI.sub("", raw.rstrip("\n")))
            if m:
                out.append((parse_ms(m.group(1), m.group(2), m.group(3)), m.group(4)))
    return out


def first(lines, rx, after_ms=None, after_idx=-1):
    """(index, ms, match) of the first line matching rx after an index and a time."""
    for i, (ms, text) in enumerate(lines):
        if i <= after_idx or (after_ms is not None and ms < after_ms):
            continue
        m = rx.search(text)
        if m:
            return i, ms, m
    return None


def count(lines, rx, after_ms=None):
    return sum(1 for ms, text in lines if (after_ms is None or ms >= after_ms) and rx.search(text))


def score(arm, seed, west, east, amf, pingpong_ms, a3_slack_ms, ue=None, window_ms=120_000):
    """One episode's row. `ue` (the UE's log) starts the window; None: no window."""
    row = dict.fromkeys(COLUMNS, "-")
    row.update(seed=str(seed), arm=arm)
    if not west or not amf:
        row.update(outcome="broken", chain="empty-log")
        return row
    deadline = None
    if ue is not None:
        start = first(ue, RE_ROUTE_START)
        if not start:
            row.update(outcome="broken", chain="route-start")
            return row
        deadline = start[1] + window_ms

    def in_window(hit):
        return hit if hit and (deadline is None or hit[1] <= deadline) else None

    # Decision Q4: no HandoverNotify in the window is a failure, whatever else.
    no_notify = not in_window(first(amf, RE_AMF_NOTIFY))
    failures = (count(west, RE_GNB_FAILURE) + count(east, RE_GNB_FAILURE) + count(amf, RE_AMF_FAILURE)
                + (1 if no_notify else 0))
    fallbacks, stranded, exhausted = (count(west, rx) for rx in (RE_FALLBACK, RE_STRANDED, RE_EXHAUSTED))
    row.update(failures=str(failures), fallbacks=str(fallbacks), stranded=str(stranded),
               exhausted=str(exhausted), pingpong="0", returns="0")
    would = first(west, RE_A3_WOULD)
    if would:
        row.update(a3_lo_ms=would[2].group(1), a3_hi_ms=would[2].group(2))

    # The trigger: the predictor's two lines, or the UE's A3 report.
    if arm in ("lsq", "onnx"):
        pred = first(west, RE_PREDICTION)
        if pred:
            row["predictor"] = pred[2].group(1)
        rec = pred and first(west, RE_RECOMMENDED, after_idx=pred[0])
        trigger = rec
        missing = "prediction" if not pred else ("recommended" if not rec else None)
    else:
        trigger = first(west, RE_A3_REPORT)
        missing = None if trigger else "a3-report"
        if trigger:
            row["a3_report_ms"] = str(trigger[1])

    chain = missing
    notify = None
    if chain is None:
        required = first(west, RE_REQUIRED, after_idx=trigger[0])
        notify = required and in_window(first(amf, RE_AMF_NOTIFY, after_ms=required[1]))
        if not required:
            chain = "handover-required"
        elif not notify:
            chain = "amf-handover-notify"
        elif not first(east, RE_SENT_NOTIFY, after_ms=required[1]):
            chain = "east-handover-notify"
    if chain is None and arm in ("lsq", "onnx"):
        lead = first(west, RE_LEAD, after_idx=trigger[0])
        if not would:
            chain = "a3-would-trigger"
        elif not lead:
            chain = "lead-time"
        else:
            row.update(lead_lo_ms=lead[2].group(1), lead_hi_ms=lead[2].group(2))
    if chain is None and arm == "a3" and not would:
        chain = "a3-would-trigger"
    chain = chain or "ok"
    row["chain"] = chain

    # Hand-backs to gnb-west after the first completed handover.
    if notify:
        backs = [ms for ms, text in west if ms >= notify[1] and RE_SENT_NOTIFY.search(text)]
        row["returns"] = str(len(backs))
        row["pingpong"] = "1" if backs and backs[0] - notify[1] <= pingpong_ms else "0"

    # The outcome: the reason a seed is not a clean chain comes first.
    if arm in ("lsq", "onnx"):
        if fallbacks:
            outcome = "fallback"
        elif stranded:
            outcome = "stranded"
        elif failures or chain in ("amf-handover-notify", "east-handover-notify"):
            outcome = "failed"
        elif chain != "ok":
            outcome = "broken"
        elif row["predictor"] != arm:
            outcome = "predictor-mislabelled"
        else:
            outcome = "predictive"
    elif arm == "a3":
        if first(west, RE_PREDICTION) or first(west, RE_RECOMMENDED):
            outcome = "predicted"
        elif failures or chain in ("amf-handover-notify", "east-handover-notify"):
            outcome = "failed"
        elif chain != "ok":
            outcome = "broken"
        else:
            lo, hi, rep = int(row["a3_lo_ms"]), int(row["a3_hi_ms"]), int(row["a3_report_ms"])
            outcome = "a3" if lo - a3_slack_ms <= rep <= hi + a3_slack_ms else "a3-outside"
    else:  # off: the base core, no demo line on either gNB
        if any(RE_ANY_DEMO.search(t) for _, t in west + east):
            outcome = "demo-line"
        elif failures or chain in ("amf-handover-notify", "east-handover-notify"):
            outcome = "failed"
        elif chain != "ok":
            outcome = "broken"
        else:
            outcome = "a3"
    row["outcome"] = outcome
    return row


CHAINED = {"lsq": {"predictive"}, "onnx": {"predictive"}, "a3": {"a3"}, "off": {"a3"}}


def p10(values):
    """The 10th percentile, nearest rank (no interpolation)."""
    s = sorted(values)
    return s[max(0, -(-len(s) // 10) - 1)]


def parse_seeds(spec):
    """"A-B" or a list separated by spaces or commas, as d441d-check.sh takes it."""
    m = re.fullmatch(r"\s*(\d+)-(\d+)\s*", spec)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(s) for s in re.split(r"[\s,]+", spec.strip()) if s]


def table(arm, rows, min_chains, seeds):
    """Markdown summary of one arm, and whether it passes.

    Every planned seed needs exactly one row, and no row may be unplanned: a
    seed that never ran must not pass as a short table."""
    n = len(rows)
    got = [int(r["seed"]) for r in rows]
    missing = sorted(set(seeds) - set(got))
    dupes = sorted({s for s in got if got.count(s) > 1})
    extra = sorted(set(got) - set(seeds))
    chained = sum(1 for r in rows if r["outcome"] in CHAINED[arm])
    los = [int(r["lead_lo_ms"]) for r in rows if r["lead_lo_ms"] != "-"]
    his = [int(r["lead_hi_ms"]) for r in rows if r["lead_hi_ms"] != "-"]

    def stat(v, f):
        return f"{f(v)}" if v else "-"

    total = {c: sum(int(r[c]) for r in rows if r[c] != "-")
             for c in ("pingpong", "returns", "failures", "fallbacks", "stranded", "exhausted")}
    outcomes = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    lines = [
        f"| arm | seeds | chain held | lead lower bound: median / p10 (ms) | lead upper bound: median (ms) "
        f"| ping-pongs | returns | failures | A3 fallbacks | stranded | exhausted |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        f"| {arm} | {n} | {chained} | {stat(los, statistics.median_low)} / {stat(los, p10)} "
        f"| {stat(his, statistics.median_low)} | {total['pingpong']} | {total['returns']} "
        f"| {total['failures']} | {total['fallbacks']} | {total['stranded']} | {total['exhausted']} |",
        "",
        "outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())),
    ]
    positive = sum(1 for v in los if v > 0)
    if los:
        lines.append(f"lead lower bound positive on {positive}/{len(los)} scored seeds")
    plan_ok = not (missing or dupes or extra)
    if not plan_ok:
        lines.append(f"seed plan broken: missing {missing}, duplicated {dupes}, unplanned {extra}")
    ok = plan_ok and n > 0 and chained >= min_chains
    lines.append(f"VERDICT {arm}: {'PASS' if ok else 'FAIL'} ({chained}/{n} chained, need {min_chains};"
                 f" {len(seeds)} seed(s) planned)")
    return "\n".join(lines), ok


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("seed\t"):
                continue
            cells = line.split("\t")
            if len(cells) != len(COLUMNS):
                raise ValueError(f"{path}:{n}: {len(cells)} columns, expected {len(COLUMNS)}")
            rows.append(dict(zip(COLUMNS, cells)))
    return rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--arm", required=True, choices=sorted(CHAINED))
    s.add_argument("--seed", required=True, type=int)
    s.add_argument("--west", required=True)
    s.add_argument("--east", required=True)
    s.add_argument("--amf", required=True)
    s.add_argument("--ue", help="the UE log: the window starts at its route start")
    s.add_argument("--window-ms", type=int, default=120_000)
    s.add_argument("--pingpong-ms", type=int, default=5000)
    s.add_argument("--a3-slack-ms", type=int, default=250)
    s.add_argument("--header", action="store_true", help="print the column header first")
    t = sub.add_parser("table")
    t.add_argument("--arm", required=True, choices=sorted(CHAINED))
    t.add_argument("--min-chains", required=True, type=int)
    t.add_argument("--seeds", required=True, help='the planned seeds, "A-B" or a list')
    t.add_argument("rows")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        row = score(a.arm, a.seed, read_log(a.west), read_log(a.east), read_log(a.amf),
                    a.pingpong_ms, a.a3_slack_ms, read_log(a.ue) if a.ue else None, a.window_ms)
        if a.header:
            print("\t".join(COLUMNS))
        print("\t".join(row[c] for c in COLUMNS))
        return 0
    try:
        rows = read_rows(a.rows)
    except (OSError, ValueError) as e:
        print(f"DEMO CHECK FAILED: {e}", file=sys.stderr)
        return 1
    bad = [r for r in rows if r["arm"] != a.arm]
    if bad:
        print(f"DEMO CHECK FAILED: {len(bad)} row(s) of another arm in {a.rows}", file=sys.stderr)
        return 1
    try:
        seeds = parse_seeds(a.seeds)
    except ValueError:
        print(f"DEMO CHECK FAILED: --seeds {a.seeds!r} is not a seed list", file=sys.stderr)
        return 1
    if not seeds or not 1 <= a.min_chains <= len(seeds):
        print(f"DEMO CHECK FAILED: --min-chains {a.min_chains} outside 1..{len(seeds)}", file=sys.stderr)
        return 1
    text, ok = table(a.arm, rows, a.min_chains, seeds)
    print(text)
    if not ok:
        print(f"DEMO CHECK FAILED: [{a.arm}] too few seeds held the chain", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
