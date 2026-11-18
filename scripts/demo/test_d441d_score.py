#!/usr/bin/env python3
"""Self-test of d441d_score.py (nextgsim#467) on canned logs. No docker, no cluster.

    python3 -I scripts/demo/test_d441d_score.py

Every outcome the scorer can give has a log that produces it, and every link
of each chain has a log missing exactly that link, which must be named. The
lines are the strings the code emits, with the two timestamp prefixes the
harnesses capture (`docker compose logs -t`, `kubectl logs --timestamps`) and
the ANSI colour codes the gNB prints.
"""
import os
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d441d_score as s  # noqa: E402

E = "\x1b[32m INFO\x1b[0m "
T0 = 1_763_460_000_000  # 2025-11-18T10:00:00Z in ms


def at(ms, text, compose=True, svc="gnb-west-1"):
    import datetime as dt
    stamp = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z"
    return f"{svc}  | {stamp} {text}" if compose else f"{stamp} {text}"


PRED = E + "nextgsim_gnb::nwdaf::task: [demo:d441d-predictive-ho] NWDAF: handover prediction ue=1 predictor={p} target=32 conf=0.98 attempt=1/3"
REC = E + "nextgsim_gnb::rrc::task: RRC: NWDAF-recommended handover for UE 1 to cell 32 (confidence 0.98)"
A3REP = E + "nextgsim_gnb::rrc::task: UE[1] measurement-triggered N2 handover to PCI 32: gNB 2, cell 0x20, TAC 1"
REQ = E + "nextgsim_gnb::ngap::task: Sent Handover Required: amf_ue_ngap_id=1, ran_ue_ngap_id=1"
WOULD = E + "nextgsim_gnb::nwdaf::task: [demo:d441d-predictive-ho] A3 would trigger t={lo}..{hi} ue=1 target=32 serving=-80 dBm neighbour=-75 dBm (the UE's A3 on its 2.5 s cycle, at an unknown phase)"
LEAD = E + "nextgsim_gnb::nwdaf::task: [demo:d441d-predictive-ho] lead time ue=1 target=32 predictive_t={pt} a3_t={lo}..{hi} lead_ms={a}..{b}"
NOTIFY_AMF = "INFO amfd::ngap: HandoverNotify from association 2 (amf_ue_ngap_id=1, ran_ue_ngap_id=1)"
NOTIFY_GNB = E + "nextgsim_gnb::ngap::task: Sent Handover Notify: amf_ue_ngap_id=1, ran_ue_ngap_id=1"
FALLBACK = "\x1b[33m WARN\x1b[0m nextgsim_gnb::rrc::task: [demo:d441d-predictive-ho] A3 fallback=a3 n=1 UE[1] PCI 32: no predictive handover within 1000 ms of its A3 report; acting on A3"
STRANDED = "WARN [demo:d441d-predictive-ho] stranded ue=1 target=32: A3 fired at t=1..2 and neither the predictor nor the A3 fallback moved the UE"
EXHAUSTED = "WARN [demo:d441d-predictive-ho] NWDAF: handover prediction exhausted ue=1 target=32: still served after 3 attempt(s); none until it is served afresh"
PREP_FAIL = E + "nextgsim_gnb::ngap::task: Handover Preparation Failure: amf_ue_ngap_id=1, ran_ue_ngap_id=1, cause=RadioNetwork(CellNotAvailable)"
BANNER = "WARN DEMO MODE ACTIVE: d441d-predictive-ho=lsq"


class Episode:
    """One seed's three logs, built line by line."""

    def __init__(self, compose=True):
        self.compose = compose
        self.logs = {"west": [], "east": [], "amf": []}

    def add(self, which, ms, text):
        svc = {"west": "gnb-west-1", "east": "gnb-east-1", "amf": "amf-1"}[which]
        self.logs[which].append(at(T0 + ms, text, self.compose, svc))
        return self

    def score(self, arm, ue=None, **kw):
        d = tempfile.mkdtemp()
        paths = {}
        for k, lines in self.logs.items():
            paths[k] = os.path.join(d, k + ".log")
            with open(paths[k], "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        return s.score(arm, 7, s.read_log(paths["west"]), s.read_log(paths["east"]),
                       s.read_log(paths["amf"]), kw.get("pingpong_ms", 5000), kw.get("a3_slack_ms", 250),
                       ue, kw.get("window_ms", 120_000))


def predictive(p="lsq", compose=True, lead=(2400, 5000)):
    e = Episode(compose)
    e.add("west", 0, BANNER).add("amf", 0, "INFO amfd: NG Setup from gNB 2")
    e.add("west", 1000, PRED.format(p=p)).add("west", 1010, REC).add("west", 1100, REQ)
    e.add("east", 1300, NOTIFY_GNB).add("amf", 1350, NOTIFY_AMF)
    lo, hi = T0 + 1000 + lead[0], T0 + 1000 + lead[1]
    e.add("west", 7000, WOULD.format(lo=lo, hi=hi))
    e.add("west", 7001, LEAD.format(pt=T0 + 1000, lo=lo, hi=hi, a=lead[0], b=lead[1]))
    return e


def a3_episode(report_at=5000, lo=4000, hi=6600):
    e = Episode()
    e.add("west", 0, BANNER.replace("=lsq", "=a3")).add("amf", 0, "INFO amfd: NG Setup from gNB 2")
    e.add("west", report_at, A3REP).add("west", report_at + 50, REQ)
    e.add("east", report_at + 200, NOTIFY_GNB).add("amf", report_at + 250, NOTIFY_AMF)
    e.add("west", hi + 100, WOULD.format(lo=T0 + lo, hi=T0 + hi))
    return e


class Timestamps(unittest.TestCase):
    def test_both_prefixes_and_ansi(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("gnb-west-1  | 2026-11-18T10:00:00.123456789Z \x1b[32mINFO\x1b[0m a\n")
            f.write("2026-11-18T10:00:01.5Z b\n2026-11-18T12:00:02+02:00 c\nno stamp\n")
        got = s.read_log(f.name)
        base = 1_794_996_000_000  # 2026-11-18T10:00:00Z
        self.assertEqual(got, [(base + 123, "INFO a"), (base + 1500, "b"), (base + 2000, "c")])


class PredictiveArm(unittest.TestCase):
    def test_the_chain_holds(self):
        for compose in (True, False):
            r = predictive(compose=compose).score("lsq")
            self.assertEqual((r["outcome"], r["chain"], r["predictor"]), ("predictive", "ok", "lsq"), r)
            self.assertEqual((r["lead_lo_ms"], r["lead_hi_ms"]), ("2400", "5000"))
            self.assertEqual((r["failures"], r["pingpong"], r["returns"]), ("0", "0", "0"))

    def test_a_negative_lead_still_scores(self):
        r = predictive(lead=(-1200, 1400)).score("lsq")
        self.assertEqual((r["outcome"], r["lead_lo_ms"]), ("predictive", "-1200"))

    def test_each_missing_link_is_named(self):
        cases = {
            "prediction": PRED.format(p="lsq"),  # the tick patched out (the revert arm)
            "recommended": REC,
            "handover-required": REQ,
            "amf-handover-notify": NOTIFY_AMF,
            "east-handover-notify": NOTIFY_GNB,
            "a3-would-trigger": "A3 would trigger",
            "lead-time": "lead time ue=",
        }
        for link, drop in cases.items():
            e = predictive()
            for k in e.logs:
                e.logs[k] = [l for l in e.logs[k] if drop not in l]
            r = e.score("lsq")
            self.assertEqual(r["chain"], link, (link, r))
            self.assertNotEqual(r["outcome"], "predictive", (link, r))

    def test_out_of_order_is_not_a_chain(self):
        e = predictive()
        # The recommendation before the prediction: RRC acted on something else.
        e.logs["west"] = [l for l in e.logs["west"] if "NWDAF-recommended" not in l]
        e.add("west", -500, REC)
        e.logs["west"].sort()
        self.assertEqual(e.score("lsq")["chain"], "recommended")

    def test_a_notify_before_the_required_is_not_this_handover(self):
        e = predictive()
        e.logs["amf"] = []
        e.add("amf", 500, NOTIFY_AMF)
        self.assertEqual(e.score("lsq")["chain"], "amf-handover-notify")

    def test_fallback_stranded_exhausted_failed(self):
        e = Episode().add("west", 0, BANNER).add("west", 5000, FALLBACK).add("west", 5010, REQ)
        e.add("amf", 5300, NOTIFY_AMF).add("east", 5200, NOTIFY_GNB)
        self.assertEqual(e.score("lsq")["outcome"], "fallback")
        e = Episode().add("west", 0, BANNER).add("west", 9000, STRANDED).add("amf", 1, "x")
        self.assertEqual(e.score("lsq")["outcome"], "stranded")
        e = Episode().add("west", 0, BANNER).add("west", 9000, EXHAUSTED).add("amf", 1, "x")
        r = e.score("lsq")
        self.assertEqual((r["outcome"], r["exhausted"], r["failures"]), ("failed", "1", "1"))
        e = predictive().add("west", 1200, PREP_FAIL)
        r = e.score("lsq")
        self.assertEqual((r["outcome"], r["failures"]), ("failed", "1"))

    def test_onnx_served_by_the_fallback_is_mislabelled(self):
        self.assertEqual(predictive(p="onnx").score("onnx")["outcome"], "predictive")
        self.assertEqual(predictive(p="fallback").score("onnx")["outcome"], "predictor-mislabelled")
        self.assertEqual(predictive(p="lsq").score("onnx")["outcome"], "predictor-mislabelled")

    def test_ping_pong_and_returns(self):
        e = predictive().add("west", 1350 + 4000, NOTIFY_GNB)
        r = e.score("lsq")
        self.assertEqual((r["pingpong"], r["returns"]), ("1", "1"))
        e = predictive().add("west", 1350 + 9000, NOTIFY_GNB).add("west", 1350 + 20000, NOTIFY_GNB)
        r = e.score("lsq")
        self.assertEqual((r["pingpong"], r["returns"]), ("0", "2"))
        self.assertEqual(predictive().add("west", 1350 + 4000, NOTIFY_GNB).score("lsq", pingpong_ms=1000)["pingpong"], "0")

    def test_an_empty_log_is_broken_not_clean(self):
        e = Episode()
        self.assertEqual(e.score("lsq")["chain"], "empty-log")


class Q4Failures(unittest.TestCase):
    """Decision Q4: no HandoverNotify within 120 s of the route start is a failure."""

    def ue(self, start_ms):
        return [(T0 + start_ms, "INFO nextgsim_ue::rls::task: Simulated motion started: sim_route from (20, 17, 0) at 40.0 s after start-up")]

    def test_nothing_handed_over_is_failed_not_broken(self):
        e = predictive()
        e.logs["west"] = [l for l in e.logs["west"] if "handover prediction" not in l]
        e.logs["amf"] = [l for l in e.logs["amf"] if "HandoverNotify" not in l]
        r = e.score("lsq")
        self.assertEqual((r["outcome"], r["chain"], r["failures"]), ("failed", "prediction", "1"), r)
        e = a3_episode()
        e.logs["west"] = [l for l in e.logs["west"] if "measurement-triggered" not in l]
        e.logs["amf"] = [l for l in e.logs["amf"] if "HandoverNotify" not in l]
        r = e.score("a3")
        self.assertEqual((r["outcome"], r["chain"], r["failures"]), ("failed", "a3-report", "1"), r)

    def test_a_notify_after_the_window_is_a_failure(self):
        # The notify lands at +1350 ms; the route started 130 s before it.
        r = predictive().score("lsq", ue=self.ue(1350 - 130_000))
        self.assertEqual((r["outcome"], r["chain"], r["failures"]), ("failed", "amf-handover-notify", "1"), r)
        r = predictive().score("lsq", ue=self.ue(1350 - 110_000))
        self.assertEqual((r["outcome"], r["failures"]), ("predictive", "0"), r)

    def test_no_route_start_is_broken(self):
        self.assertEqual(predictive().score("lsq", ue=[])["chain"], "route-start")

    def test_a_rejected_notify_is_a_failure(self):
        e = predictive().add("amf", 1340, "WARN amfd::ngap_path: HandoverNotify from association 2 rejected: NoUeContext")
        r = e.score("lsq")
        self.assertEqual((r["outcome"], r["failures"]), ("failed", "1"), r)


class A3Arm(unittest.TestCase):
    def test_the_report_inside_the_replica_range(self):
        r = a3_episode().score("a3")
        self.assertEqual((r["outcome"], r["chain"]), ("a3", "ok"), r)
        self.assertEqual(r["a3_report_ms"], str(T0 + 5000))

    def test_outside_the_range(self):
        self.assertEqual(a3_episode(report_at=7000).score("a3")["outcome"], "a3-outside")
        self.assertEqual(a3_episode(report_at=6800).score("a3")["outcome"], "a3")  # within the slack
        self.assertEqual(a3_episode(report_at=3700).score("a3", a3_slack_ms=100)["outcome"], "a3-outside")

    def test_a_predictor_line_in_the_a3_arm(self):
        e = a3_episode().add("west", 100, PRED.format(p="lsq"))
        e.logs["west"].sort()
        self.assertEqual(e.score("a3")["outcome"], "predicted")

    def test_missing_links(self):
        for link, drop in {"a3-report": "measurement-triggered", "handover-required": "Sent Handover Required",
                           "amf-handover-notify": "HandoverNotify from", "a3-would-trigger": "A3 would trigger"}.items():
            e = a3_episode()
            for k in e.logs:
                e.logs[k] = [l for l in e.logs[k] if drop not in l]
            self.assertEqual(e.score("a3")["chain"], link, link)


class OffArm(unittest.TestCase):
    def off(self):
        e = Episode().add("west", 5000, A3REP).add("west", 5050, REQ)
        return e.add("east", 5200, NOTIFY_GNB).add("amf", 5250, NOTIFY_AMF)

    def test_the_base_handover(self):
        r = self.off().score("off")
        self.assertEqual((r["outcome"], r["chain"]), ("a3", "ok"), r)

    def test_any_demo_line_fails(self):
        self.assertEqual(self.off().add("east", 1, BANNER).score("off")["outcome"], "demo-line")
        self.assertEqual(self.off().add("west", 9000, WOULD.format(lo=1, hi=2)).score("off")["outcome"], "demo-line")


class Table(unittest.TestCase):
    def rows(self, outcomes, leads):
        rows = []
        for i, (o, lead) in enumerate(zip(outcomes, leads), 1):
            r = dict.fromkeys(s.COLUMNS, "0")
            r.update(seed=str(i), arm="lsq", outcome=o, chain="ok", predictor="lsq",
                     lead_lo_ms=str(lead) if lead is not None else "-",
                     lead_hi_ms=str(lead + 2600) if lead is not None else "-",
                     a3_lo_ms="-", a3_hi_ms="-", a3_report_ms="-")
            rows.append(r)
        return rows

    def test_median_p10_and_verdict(self):
        leads = [1000 * i for i in range(1, 21)]
        text, ok = s.table("lsq", self.rows(["predictive"] * 20, leads), 18, list(range(1, 21)))
        self.assertTrue(ok)
        self.assertIn("| lsq | 20 | 20 | 10000 / 2000 | 12600 |", text)
        self.assertIn("lead lower bound positive on 20/20", text)

    def test_too_few_chains_fail(self):
        outcomes = ["predictive"] * 17 + ["fallback", "stranded", "failed"]
        text, ok = s.table("lsq", self.rows(outcomes, [1000] * 17 + [None] * 3), 18, list(range(1, 21)))
        self.assertFalse(ok)
        self.assertIn("VERDICT lsq: FAIL (17/20 chained, need 18;", text)
        self.assertIn("fallback=1", text)

    def test_no_rows_fail(self):
        self.assertFalse(s.table("lsq", [], 1, [1])[1])

    def test_the_seed_plan_is_enforced(self):
        plan = list(range(1, 25))
        rows = self.rows(["predictive"] * 22, [1000] * 22)       # seeds 1..22 of 24
        text, ok = s.table("lsq", rows, 22, plan)
        self.assertFalse(ok)
        self.assertIn("missing [23, 24]", text)
        twice = self.rows(["predictive"] * 12, [1000] * 12) * 2   # seeds 1..12, twice
        text, ok = s.table("lsq", twice, 22, plan)
        self.assertFalse(ok)
        self.assertIn("duplicated [1, 2, 3", text)
        rows = self.rows(["predictive"] * 3, [1000] * 3)
        text, ok = s.table("lsq", rows, 1, [1, 2])
        self.assertFalse(ok)
        self.assertIn("unplanned [3]", text)
        self.assertTrue(s.table("lsq", rows, 1, [1, 2, 3])[1])

    def test_parse_seeds(self):
        self.assertEqual(s.parse_seeds("1-4"), [1, 2, 3, 4])
        self.assertEqual(s.parse_seeds("1 2,3"), [1, 2, 3])

    def test_cli_round_trip_and_malformed_rows(self):
        d = tempfile.mkdtemp()
        e = predictive()
        paths = []
        for k, lines in e.logs.items():
            p = os.path.join(d, k + ".log")
            with open(p, "w") as f:
                f.write("\n".join(lines) + "\n")
            paths.append(p)
        out = os.path.join(d, "rows.tsv")
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(s.main(["score", "--arm", "lsq", "--seed", "3", "--west", paths[0],
                                     "--east", paths[1], "--amf", paths[2], "--header"]), 0)
        with open(out, "w") as f:
            f.write(buf.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(s.main(["table", "--arm", "lsq", "--seeds", "3", "--min-chains", "1", out]), 0)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            # MINOR 6: --min-chains outside 1..K is refused, never a vacuous pass.
            self.assertEqual(s.main(["table", "--arm", "lsq", "--seeds", "3", "--min-chains", "0", out]), 1)
            self.assertEqual(s.main(["table", "--arm", "lsq", "--seeds", "3", "--min-chains", "2", out]), 1)
            self.assertEqual(s.main(["table", "--arm", "lsq", "--seeds", "2", "--min-chains", "1", out]), 1)
        with open(out, "a") as f:
            f.write("1\tlsq\tpredictive\n")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(s.main(["table", "--arm", "lsq", "--seeds", "3", "--min-chains", "1", out]), 1)
            self.assertEqual(s.main(["table", "--arm", "a3", "--seeds", "3", "--min-chains", "1", out]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
