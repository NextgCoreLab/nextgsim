# D440A: the gNB predicts where an idle UE went, and the AMF pages there first

Demo key `d440a-ai-paging`, ticket
[nextgsim#462](https://github.com/NextgCoreLab/nextgsim/issues/462), program
anchor [nextgsim#458](https://github.com/NextgCoreLab/nextgsim/issues/458).
The plan is `specs/demo-d440a-ai-paging-plan-462.md` in the coordinator workspace.
The code ships in these PRs:

| PR | Repo | What |
|---|---|---|
| [#514](https://github.com/NextgCoreLab/nextgsim/issues/514) | nextgsim | base fix: the UE acts on PCCH only from its serving cell (TS 38.304 §5.2.5), without which page-all and a targeted page measure the same |
| [nextgcore#1442](https://github.com/NextgCoreLab/nextgcore/issues/1442) | nextgcore | base feature: amfd stores the Recommended Cells and RAN Nodes for Paging IE and, under `paging.strategy: recommended-first`, pages those nodes first (TS 23.502 §4.2.3.3) |
| 3a, [#525](https://github.com/NextgCoreLab/nextgsim/pull/525) | nextgsim | `sim_route` waypoints, the paging grid, the M0 gate (idle reselection, MT Service Request through another gNB) |
| 3b, [#535](https://github.com/NextgCoreLab/nextgsim/pull/535) | nextgsim | the predictors (`last-served`, `ct`, `lsq`, `onnx`), the IE encoder, `nr-cli` set/show, the seeded replay and its table |
| 3c, [#584](https://github.com/NextgCoreLab/nextgsim/pull/584) | nextgsim | the six-gNB compose harness and its live arms, folded into the nightly `demo-d440a` job |
| 4, [nextgcore#1783](https://github.com/NextgCoreLab/nextgcore/pull/1783) | nextgcore | the amfd demo layer (modes, demo lines, `amf_paging_attempts_total`) and the devtest1 EKS overlay |
| 5 | nextgsim | this page |

## Purpose

When the gNB releases a UE to CM-IDLE, it predicts where the UE is heading and
puts that prediction into the UE CONTEXT RELEASE COMPLETE. The prediction is an
*Information on Recommended Cells and RAN Nodes for Paging* (TS 38.413
§9.3.1.100):
- the serving gNB first, as visited;
- then the predicted gNB, as non-visited.

When downlink data arrives, amfd pages those gNBs first. Only if the UE does not
answer does it widen to the whole registration area. The aim is fewer PAGING
messages per mobile-terminated call, at the same reachability.

- **Success signal, in order:**
  1. the serving gNB logs `[demo:d440a-ai-paging] paging recommendation ... predictor=<mode>`;
  2. then `UE Context Release Complete sent ... nodes=[0x<serving>(visited),...]`;
  3. amfd logs `Paging sent to K/M gNB(s) recommended by the RAN in the registration area ... attempt 1`;
  4. then `[demo:d440a-ai-paging] page answered ue=<supi> via gNB 0x.. attempt 1`.
- **What the spec says:**
  - TS 38.413 §8.3.3.2: the AMF "shall, if supported, store it and may use it for subsequent paging".
  - TS 23.502 §4.2.3.3: the AMF "shall take that information into account to determine the (R)AN nodes for paging".
  - Storing and using the IE is therefore a base amfd feature (`paging.strategy: recommended-first`), not demo code. The demo adds only the predictors, the log lines and the counter.

## How it works

### Components
- **gNB** (`--features demo-d440a-ai-paging`):
  - The NWDAF task keeps each UE's heartbeat position history.
  - At release, the NGAP task asks it for a recommendation in the configured mode:

    | Mode | Recommends |
    |---|---|
    | `page-all` | nothing (the IE is not sent) |
    | `last-served` | the serving gNB only |
    | `ct` | a per-axis CV Kalman filter, a least-squares turn-rate fit on the filtered heading, and a constant-turn arc extrapolation |
    | `lsq` | a least-squares straight-line extrapolation |
    | `onnx` | the verified D440B model |

  - The predicted position resolves to the nearest gNB of the paging grid.
- **amfd** (base, `paging.strategy: recommended-first`):
  - stores the IE on the UE context at Release Complete;
  - copies it into the paging context;
  - pages the recommended gNBs within the registration area on attempt 1, and the whole area on every T3513 retransmission.
- **amfd demo layer** (`--features demo-d440a-ai-paging` plus the `demo:` block):
  - It accepts the gNB's modes, so one block names the arm in every NF.
  - Every mode forces `recommended-first`, and amfd logs that at startup.
  - It adds the `[demo:d440a-ai-paging] Paging sent ... scope= targets= attempt` and `page answered ... via gNB` lines, and `amf_paging_attempts_total{scope}`.

### The grid
Six gNBs on one TAC, 120 m apart, so the registration area holds all six.

| gNB | NCI | position |
|---|---|---|
| 1 | 0x010 | (0, 0) |
| 2 | 0x020 | (120, 0) |
| 3 | 0x030 | (240, 0) |
| 4 | 0x040 | (0, 120) |
| 5 | 0x050 | (120, 120) |
| 6 | 0x060 | (240, 120) |

Each gNB lists the other five as handover neighbours and as its paging grid. The
configs are generated from `config/demo-ai-paging/d440a-scenario.yaml` by
`nwdaf::d440a::replay::live`, and a test fails on any drift:
- `config/demo-ai-paging/gnb-{1..6}.yaml`;
- `ue-seed-{1..5}.yaml`;
- `d440a-live-seeds.yaml`.

### A live seed
- The UE registers, starts its PDU session, and 60 s later rides the seed's route, which is the replay's route for that seed.
- The harness releases it at `release_at_s` on whichever gNB serves it then.
- At `page_at_s` the UPF pings it, so the SMF's N1N2MessageTransfer makes amfd page it.

## How to run it

### Compose (nextgsim)
```bash
# nextgsim-gnb:latest built with --features demo-d440a-ai-paging; nr-cli in binaries/
NEXTGCORE_DIR=../nextgcore scripts/demo/d440a-check.sh <off|page-all|last-served|ct|lsq> <seed> [revert-amf]
```
The nightly `demo-d440a` job runs seed 1 in every mode, plus both revert arms:
- `revert-amf`: amfd on `registration-area`;
- `reverts/d440a-no-ie.patch`: the gNB predicts but sends no IE.

Both must fail at `[amf] step 1/2 missing`. The dispatch input `d440a_seeds` runs all five seeds.

### EKS (devtest1)
The overlay is nextgcore `deploy/eks/overlays/demo-d440a-ai-paging/`. It is
ClusterIP only and publishes nothing. Its grid is generated from the configs
above, with Service names in place of the compose addresses. The operator runs
the arms:

```bash
# Trees side by side (deploy-eks.sh renders ../nextgsim beside nextgcore).
BASE_CORE_IMAGE_TAG=<an existing post-#1752 -6g core tag> \
  deploy/eks/overlays/demo-d440a-ai-paging/build-images.sh   # prints the tags
for mode in ct last-served lsq page-all off; do
  CORE_IMAGE_TAG=<tag> SIM_IMAGE_TAG=<tag> SIM_SHA=<sha> YES=true \
    flock /tmp/eks-nextg-demo.lock k8s/demo-check.sh d440a-ai-paging "$mode"
done
DEMO_VARIANT=revert ... k8s/demo-check.sh d440a-ai-paging ct   # must FAIL at [amf]
```

`analyze.py` checks, in order:
- the UE chain;
- amfd's base scope phrase and its demo lines;
- the serving gNB's recommendation, with attempt 1's targets equal to the recommended nodes in order;
- every gNB's banner and grid;
- the counter;
- the ping.

It records the measurement in `scenario/result.txt`.

## Results

### Pass/fail is the chain, not the hit
Each arm passes when its whole chain holds. Whether attempt 1 reached the gNB the
UE answered through is the **measurement**: it is recorded, and it is never a pass
condition. The replay table is the statistical result, and the live runs show
that the chain behaves the same through real NGAP, a real amfd and real radio
timing.

### The replay (in-process, deterministic)
From `config/demo-ai-paging/d440a-replay-table.txt`, which is written by the
replay and asserted exactly. Here are the 2000-seed rows. "PAGINGs" counts every
PAGING message of a mobile-terminated call, and a missed first attempt costs six
more.

| family | mode | hit rate (95% interval) | mean PAGINGs |
|---|---|---|---|
| any | page-all | 100.0% | 6.00 |
| constant-turn | last-served | 56.4% (54.2–58.5) | 3.62 |
| constant-turn | ct | 93.2% (92.0–94.2) | 1.88 |
| constant-turn | lsq | 86.0% (84.4–87.5) | 2.28 |
| constant-turn | onnx | 63.5% (61.4–65.6) | 3.28 |
| straight | ct | 95.5% (94.6–96.4) | 1.73 |
| straight | lsq | 97.1% (96.3–97.7) | 1.64 |
| turn-change | ct | 89.0% (87.5–90.3) | 2.13 |
| turn-change | lsq | 90.0% (88.7–91.3) | 2.04 |

The McNemar test is ct vs lsq, on the seeds that only one of them hit:
- ct wins on constant-turn routes (p = 2.9e-23), which are its own motion model;
- lsq wins on straight routes (p = 1.6e-6), which are its own;
- neither wins on turn-change routes (p = 0.14).

So each predictor is best on the motion it models. The claim that holds in every
family is that both predictors cut PAGINGs per call from 6.00 to about 1.6–2.3
against page-all. Across families the hit rates differ by a few points, not more.

### Live, compose (nightly harness, seed 1)
Dispatch [44272894118](https://github.com/NextgCoreLab/nextgsim/actions/runs/44272894118)
on #584's head d930937, 2026-11-18. Every chain holds, and both revert arms fail
at `[amf] step 1/2 missing`.

| mode | serving | recommended | answered (NCI) | attempt-1 hit | PAGINGs | replay, same seed |
|---|---|---|---|---|---|---|
| off | 0x6 | (no IE) | 0x60 | yes | 6 | — |
| page-all | 0x6 | (no IE) | 0x60 | yes | 6 | hit, 6 |
| last-served | 0x6 | [0x6(visited)] | 0x60 | yes | 1 | hit, 1 |
| ct | 0x6 | [0x6(visited), 0x5] | 0x60 | yes | 2 | hit, 1 (recommended [6]) |
| lsq | 0x6 | [0x6(visited)] | 0x60 | yes | 1 | hit, 1 |

This is one seed, so it is one data point per mode, not a rate. ct live also named
0x5, where the replay named only 0x6. The two can differ because the live run has
heartbeat jitter and real NGAP timing, which the replay does not model.

### Live, devtest1 EKS (seed 1)
The operator authorized this run on 2026-11-18. It ran from nextgcore 5d5460fb
(#1783) and nextgsim 88d6b9c (#584), under `flock /tmp/eks-nextg-demo.lock`,
one arm per hold. Every arm met its expectation, and every chain holds. The
revert arm fails at `[amf] step 1/2` only.
- Source: `/tmp/eks-val/demos/d440a-ai-paging/<arm>/latest/verdict.txt` and `scenario/result.txt`.
- Images: `demo-d440a-ai-paging-5d5460fb-88d6b9c1`. amf, gnb and ue were built by
  `build-images.sh`. The other core NFs carry the base tag
  `demo-d441d-predictive-ho-36f2b6b0-20c21bf2`.
- The UE was released 22.1–23.5 s into each run (`release_offset_s`).

| arm | run | verdict | serving | recommended | attempt-1 targets | answered | attempt-1 hit | PAGINGs |
|---|---|---|---|---|---|---|---|---|
| ct | 20261118T185021Z | PASS | gnb-6 | [6] | [6] | 0x6 | yes | 1 |
| last-served | 20261118T185958Z | PASS | gnb-6 | [6] | [6] | 0x6 | yes | 1 |
| lsq | 20261118T190932Z | PASS | gnb-6 | [6] | [6] | 0x6 | yes | 1 |
| page-all | 20261118T191915Z | PASS | gnb-6 | (no IE) | [3, 4, 6, 5, 1, 2] | 0x6 | yes | 6 |
| off | 20261118T192954Z | PASS | gnb-6 | (no IE) | (demo off) | — | — | 6 |
| ct + revert (must FAIL at `[amf]`) | 20261118T193934Z | FAIL, as required | gnb-6 | (no IE) | (demo off) | — | — | 6 |

The revert arm's record: `scenario/analyze.txt` says "VERDICT: REVERT HOLDS:
fails at [amf] step 1/2 only, every other check holds". demo-check's only FAIL
line is `no nextg_demo_enabled{demo=d440a-ai-paging,mode=ct} series`, because
amfd's demo is off. Teardown was ok on every arm.

This is one seed, so one data point per mode, not a rate. On EKS, ct
recommended only 0x6, where compose ct also named 0x5. Both still hit on
attempt 1.

## Honesty labels and limitations
- **Non-normative inputs.** The positions the predictors use are simulator ground truth carried on RLS heartbeats. The prediction is this simulator's research layer, not a 3GPP procedure. Only the IE, its storage and its use by the AMF are standard (TS 38.413 §9.3.1.100, TS 23.502 §4.2.3.3).
- **Out of scope:** TS 38.300 §16.20.2: "AI/ML algorithms and models are out of 3GPP scope." The predictors here are this simulator's own algorithms. Only the information they produce crosses a standard interface: the IE on NG.
- **`onnx` is not ML.** The reference D440B model is a one-step constant-velocity graph (about 1 s ahead). Its row shows the verified loading path working, not a learned predictor. It has no live arm, because that would need the ONNX Runtime in the gNB image. Its replay row comes from the CI model-verify job.
- **The replay does not model** amfd itself, NGAP round trips, connected-mode handovers, heartbeat jitter or T3513 delay. The live arms do, which is why their per-seed hit is recorded next to the replay's rather than compared against it.
- **Families are not like-for-like.** A route that leaves coverage is redrawn within its own family. Compare modes within a family.
