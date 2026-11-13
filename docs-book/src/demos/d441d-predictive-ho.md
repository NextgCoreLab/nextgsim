# D441D: predictive N2 handover vs A3

Demo key `d441d-predictive-ho` (nextgsim#467, program #458). Modes: `a3` (the
baseline), `lsq` (the least-squares predictor) and `onnx` (a d440b-verified
model), all on the same route and the same shadowing seeds.

## Purpose

An in-gNB mobility predictor triggers a real N2 handover through amfd earlier
than A3 would. It is measured as a seeded A/B against A3 on the same route and
the same shadowing seeds.

- **Success signal:** per seed, the lead time at the source gNB (the
  `predictive trigger` against the shadow `A3 would trigger`), each followed by
  amfd `HANDOVER NOTIFY` for the UE from the target gNB. A committed table
  reports lead time, ping-pongs and failures per arm.
- **Not the point:** this is not a claim that the predictor is accurate in
  general, and not a RIC or xApp demo. The core sees the same NGAP cause
  (`HandoverDesirableForRadioReason`) in both arms, so handover timing is the
  only evidence at the core.

## How it works

```mermaid
sequenceDiagram
  participant UE
  participant gA as gNB-A (source)
  participant AMF as amfd
  participant gB as gNB-B (target)
  UE->>gA: RLS heartbeat (sim_pos), ack carries dbm
  UE->>gA: MeasurementReport (A3)
  gA->>gA: NWDAF tick: predict (lsq | onnx), shadow A3
  gA->>gA: RRC execute_handover (lsq, onnx) or on the A3 report (a3)
  gA->>AMF: HANDOVER REQUIRED
  AMF->>gB: HANDOVER REQUEST
  gB-->>AMF: HANDOVER REQUEST ACKNOWLEDGE
  AMF->>gA: HANDOVER COMMAND
  gA->>UE: RRCReconfiguration (reconfigurationWithSync)
  UE->>gB: RRCReconfigurationComplete
  gB->>AMF: HANDOVER NOTIFY
  UE->>gA: RLS heartbeats go on: gA's shadow A3 still runs
```

The UE follows a curved `sim_route` (#462) from gnb-west toward gnb-east. Each
gNB adds seeded log-normal shadowing (#467 PR 1) to the level in its heartbeat
ack. Both the UE's A3 evaluation and the gNB's NWDAF feed read that one value.

Every heartbeat reaches the source gNB's NWDAF task. It carries the UE's
position and this cell's level at the UE. Two evaluators run on it:

- **The predictor** runs in modes `lsq` and `onnx`, once a second, for each UE
  this gNB's RRC holds. It fires when all of these hold:
  - the trajectory prediction 2 s ahead is mobility: confidence above 0.7, and
    moving. This is the least-squares fit of #467 PR 2, or the verified model;
  - the measured serving level is falling;
  - at the predicted position, a positioned neighbour beats this cell by the
    same A3 margin the UE is configured with (`a3-Offset` plus hysteresis,
    TS 38.331 §5.5.4.4), under the distance model.

  The predictor knows the neighbour positions and the distance model. It does
  not know the shadowing field. Its recommendation goes to RRC on the existing
  NWDAF path (`handle_nwdaf_handover` → `execute_handover`), and from there as
  HANDOVER REQUIRED.
- **The shadow A3 evaluator** runs in every mode. It checks inequality A3-1
  between this cell's level and each neighbour's, held for `timeToTrigger`
  (640 ms). The neighbour's level is the one the UE measures: the same shadowed
  distance model, keyed by the neighbour's NCI, with the same `shadowing`
  block (every gNB of a run uses one).

  It keeps running after a predictive handover, because the UE still
  heartbeats this cell. That is how the A3 time is known even when the
  prediction came first. It is a gNB-side replica of the UE's A3 entering
  condition at heartbeat cadence (1 s), not the UE's report itself.

The guards:

- **Ping-pong:** a UE served for less than 5 s is not predicted away.
- **Retries:** a UE still served 3 s after an attempt is tried again, at most
  3 times. Then one `exhausted` line is logged, and nothing more until the UE
  is served again.

In modes `lsq` and `onnx`, A3 is in shadow: an A3 MeasurementReport naming a
neighbour is logged and not acted on. In mode `a3`, and with the demo off, the
#162 A3 path is unchanged.

### Components and toggles

| What | Where |
|---|---|
| decision core (predictor, shadow A3, guards) | `nextgsim-gnb/src/nwdaf/d441d.rs` |
| tick, measurement feed, `nwdaf-status` section | `nextgsim-gnb/src/nwdaf/task.rs` |
| A3 in shadow, served-UE report | `nextgsim-gnb/src/rrc/task.rs` |
| registration, startup checks | `nextgsim-gnb/src/demo.rs` |
| configs | `config/demo-predictive-ho/{gnb-west,gnb-east,ue}.yaml` |

The demo follows the #458 toggle convention:

- **Gates.** The cargo feature `demo-d441d-predictive-ho` (which implies `6g`
  and `model-registry`) and the config block are both off by default:

  ```yaml
  demo:
    enabled: true
    d441d-predictive-ho:
      enabled: true
      mode: lsq        # a3 | lsq | onnx
  ```

- **Startup.** The demo logs `WARN DEMO MODE ACTIVE: d441d-predictive-ho=<mode>`
  and `[demo:d441d-predictive-ho] predictive handover mode=...` with its
  parameters.
- **Refusals.** Each of these fails startup:
  - an unknown mode (`DemoConfigError::UnknownMode`);
  - a block in a build without the feature (`UnknownKey`);
  - `onnx` without `d440b-model-verify` in mode `verify` (`MissingDependency`);
  - `nwdaf_enabled: false`;
  - no `handover_neighbours` entry with a `position`;
  - another handover-triggering demo on the same gNB (`HandoverDemoConflict`,
    for example with D446A).
- **Runtime.** `nr-cli <gnb> demo d441d-predictive-ho set mode=<mode>` /
  `show` / `demo metrics` (`nextg_demo_enabled`). A switch applies from the next
  tick and is audited.
- **Status.** `nr-cli <gnb> nwdaf-status` gains a `handover` section: mode,
  served UEs, predictions, exhausted, A3 triggers, and the last lead times.
- **Shadowing** is a separate base block, identical in every mode.

### The log chain (strings the code emits)

Predictive arm, at the source gNB, then amfd (the amfd string is the one
`scripts/demo/n2ho-check.sh` asserts):

```text
[demo:d441d-predictive-ho] NWDAF: handover prediction ue=<id> predictor=lsq target=32 conf=0.98 attempt=1/3
RRC: NWDAF-recommended handover for UE <id> to cell 32 (confidence 0.98)
Sent Handover Required: amf_ue_ngap_id=..., ran_ue_ngap_id=...
(amfd) HandoverNotify from association <target> (amf_ue_ngap_id=..., ran_ue_ngap_id=...)
[demo:d441d-predictive-ho] A3 would trigger t=<ms> ue=<id> target=32 serving=<dBm> dBm neighbour=<dBm> dBm
[demo:d441d-predictive-ho] lead time ue=<id> target=32 predictive_t=<ms> a3_t=<ms> lead_ms=<ms>
```

In the A3 arm, the handover follows the UE's A3 MeasurementReport
(`measurement-triggered N2 handover to PCI 32`). There is no `NWDAF-recommended`
line, and the shadow `A3 would trigger` line calibrates the evaluator against
the real report.

## How to run it

Build the gNB with `--features demo-d441d-predictive-ho` and run nextgcore's
two-gNB stack (`docker/rust/docker-compose.demo-n2ho.yml`, #456). Use:

- gnb-west on `config/demo-predictive-ho/gnb-west.yaml`;
- gnb-east on `config/demo-predictive-ho/gnb-east.yaml`;
- the UE on `config/demo-predictive-ho/ue.yaml`.

For each seed, set `shadowing.seed` to the same value in both gNB configs. For
each arm, change only `mode`. Both stacks publish no port.

The seeded live harness (`scripts/demo/d441d-*.sh`, K seeds per arm, with the
ping-pong and failure counts) and its CI job are the next PR. The EKS overlay
is nextgcore's part, PR 4.

## Results

**Not yet run** on the live stack or EKS. The table (≥ 20 seeds: median and
p10 lead time, ping-pongs and failures per arm) and the captured log chain go
here after the runs, as captured.

What exists now is an offline replay of the decision core:
`nwdaf::d441d::tests::the_seeded_replay_leads_a3_on_most_seeds`. It replays
the committed route at heartbeat cadence through gnb-west's engine, with the
UE handed away at the prediction, on seeds 1..=24 at σ 4 dB. It is a positive
control for the logic, not a result of the demo:

- the predictor triggers once on every seed;
- the lead is positive on every seed: 1 to 4 s, median 3 s;
- the A3 time moves with the seed.

On 100 further seeds (25..=124), the lead was positive on 100 at σ 4 dB (median
2 s), and on 96 at σ 8 dB (median 2 s, minimum 0). It does not model amfd, the
NGAP round trip or a ping-pong at the target.

## Honesty labels and limitations

- Positions are RLS `sim_pos`: simulator ground truth, not UE-reported.
- Levels are the distance model (−1 dBm per metre) plus seeded log-normal
  shadowing. Shadowing is a non-normative simulator model, and seeds are
  committed before a run.
- The baseline is the gNB's existing A3, with the offset and hysteresis from
  config.
- The predictor is a least-squares fit or a #463-verified model, labelled as
  such: `predictor=fallback` when no verified model served the prediction.
- The prediction-to-handover trigger is gNB implementation. AI/ML models are
  outside 3GPP scope (TS 38.300 §16.20).
- The shadow A3 is a replica of the UE's condition at 1 s cadence, so its time
  can differ from the UE's report by up to one heartbeat.
- Lead time is measured at the source gNB, because the core cannot tell the
  arms apart. It is not asserted positive on every seed.
- A lead is not free. A UE handed over before A3 lands where the source is
  still the stronger cell. The ping-pong and failure columns are where that
  shows.
