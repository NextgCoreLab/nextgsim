# D441E: MTLF→AnLF model provisioning with an accuracy-triggered retrain

Demo key `d441e-model-retrain`, ticket
[nextgsim#468](https://github.com/NextgCoreLab/nextgsim/issues/468), program
anchor [nextgsim#458](https://github.com/NextgCoreLab/nextgsim/issues/458).
The code is in nextgcore nwdafd and lands in five PRs:

- PR 1, [nextgcore#1454](https://github.com/NextgCoreLab/nextgcore/pull/1454) (merged): the
  MTLF's versioned model store;
- PR 2, [nextgcore#1464](https://github.com/NextgCoreLab/nextgcore/pull/1464) (merged): the AnLF client and its verification;
- PR 3, [nextgcore#1473](https://github.com/NextgCoreLab/nextgcore/pull/1473) (merged): MLModelMonitor;
- PR 4, [nextgcore#1491](https://github.com/NextgCoreLab/nextgcore/pull/1491) (merged): the retrain and the seeded series;
- PR 5, [nextgcore#1495](https://github.com/NextgCoreLab/nextgcore/pull/1495) (merged): the compose stack, the check and revert arms, the EKS
  overlay and the CI job.

> **Status:** every part is built. The results below are captured output, from
> the runs named there.

## Purpose

To show the TS 23.288 §6.2A/§6.2E split working between two real NWDAF
instances. An MTLF provisions a verified model to an AnLF, the AnLF reports
accuracy, and a measured drop triggers retraining and re-provisioning of v2.
The whole cycle is observed in CI: nextgcore's nightly `demo-d441e` job runs
the check script below.

- **Success signal:** AnLF `model v1 verified sha256=<..> from <mLModelUrl>`,
  then MTLF `accuracy 0.6x < 0.70: retrain requested`, then AnLF
  `model v2 verified ...`, with the AnLF's prediction error back under the
  threshold after v2.
- **Not the point:**
  - a claim that the model is good: it is OLS in a single-LinearRegressor
    ONNX;
  - a gNB consumer: a gNB is not an AnLF.

## How it works

```mermaid
sequenceDiagram
  participant C as Consumer (curl)
  participant AnLF as nwdafd (role anlf)
  participant NRF as nrfd
  participant MTLF as nwdafd (role mtlf)
  AnLF->>NRF: Nnrf_NFDiscovery (NWDAF, nnwdaf-mlmodelprovision)
  AnLF->>MTLF: Nnwdaf_MLModelProvision_Subscribe (NF_LOAD, suppFeats ModelProvisionExt, mlAccuracyThreshold 70)
  MTLF-->>AnLF: 201 (suppFeats 8)
  MTLF-->>AnLF: Nnwdaf_MLModelProvision_Notify (mLFileAddr.mLModelUrl v1, modelUniqueId)
  AnLF->>MTLF: GET models/NF_LOAD/{sha256} and .manifest.json
  AnLF->>AnLF: verify sha256, opset, inputs/outputs, golden; load
  AnLF->>MTLF: Nnwdaf_MLModelMonitor_Register (modelId = modelUniqueId)
  MTLF->>NRF: Nnrf_NFDiscovery (the AnLF, nnwdaf-mlmodelmonitor)
  MTLF->>AnLF: Nnwdaf_MLModelMonitor_Subscribe (accuThreshold 70, PERIODIC 30 s)
  AnLF-->>MTLF: Nnwdaf_MLModelMonitor_Notify (mlModelAcc, every 30 s)
  Note over MTLF: rolling accuracy < 0.70 after the scripted change: AR(5) refit, publish v2
  MTLF-->>AnLF: Nnwdaf_MLModelProvision_Notify (v2)
  AnLF->>MTLF: GET v2, verify, load
  C->>AnLF: Nnwdaf_AnalyticsInfo (NF_LOAD)
```

### Components

| Piece | Where | What it does |
|---|---|---|
| Model version store | nwdafd `model_store` (PR 1, ungated) | Every provisioned model is an immutable version addressed by its sha256 at `.../models/{event}/{sha256}`, with a nextgsim#463 manifest at `<url>.manifest.json`. A new version is published when the MTLF's predictor changes, and each subscription is notified once per version (TS 23.288 §6.2E.2 step 13). |
| ModelProvisionExt | nwdafd MTLF (PR 2, ungated) | TS 29.520 Table 5.4.8-1 feature 4, negotiated by `suppFeats` (TS 29.500 §6.6.2). With it, the MTLF keeps `mlEvRepCon.mlAccuracyThreshold` (0..100) and notifies `modelUniqueId` (the first 48 bits of the sha256). The NF profile declares `supportedFeatures: "8"`. |
| `nwdaf.role` | nwdafd (PR 2, ungated) | `both` (default, unchanged), `mtlf` or `anlf`. An `anlf` instance neither advertises nor routes `nnwdaf-mlmodelprovision`. An unknown role fails startup. |
| AnLF client | nwdafd `model_retrain` (PR 2, gated) | On `role: anlf` with the demo on: finds the MTLF (`nwdaf.anlf.mtlf_uri`, else NRF discovery), subscribes, and takes the Notify on `/nnwdaf-callback/v1/mtlf-model-notify/{token}` on the existing SBI listener. It downloads the model and manifest and verifies them with nwdafd's own reimplementation of #463's checks: the same manifest shape, sha256, opset and graph inputs/outputs, and every golden vector computed as the unclamped float32 `LinearRegressor` output. No ONNX Runtime runs; this is not nextgsim's `register_verified` path. A verified model becomes the NF_LOAD predictor, and a refused one leaves the current model in place. Downloads come only from the subscribed MTLF's host and port, each GET within 10 s and 1 MiB. Loads apply in Notify order. |
| MLModelMonitor | nwdafd `model_monitor` (PR 3, gated) | The MTLF takes `POST/DELETE /nnwdaf-mlmodelmonitor/v1/registrations` and subscribes at the AnLF, which produces `/subscriptions` and sends `MLModelMonitorNotify` every 30 s: accuracy `1 - mean |error|` of a one-step backtest over the newest 20 points. The MTLF feeds each report to its `FeedbackRegistry` (rolling window of 3). 404 with the block off. |
| Retrain and seeded series | nwdafd `model_refit` (PR 4, gated) | Mode `retrain`: when a registration of the active model falls below its threshold, an AR(5) least-squares refit on the newest 48 points (3/4 train, 1/4 held out) becomes the predictor and is published and notified as v2. Both NWDAFs generate the same synthetic series from unix time: per 600 s cycle, calm `0.5 + 0.1 sin` for 120 s, then alternating `0.5 ± 0.3`, noise ±0.02. |
| Compose stack and check | nextgcore `docker/rust/docker-compose.demo-d441e.yml`, `docker/rust/configs/demo-d441e/` (PR 5) | An NRF and the two NWDAFs, nothing published. `check.sh <arm>` and `reverts.sh`; `chain.py` asserts the ordered chain. |
| EKS overlay | nextgcore `deploy/eks/overlays/demo-d441e-model-retrain/` (PR 5) | `nwdaf-mtlf` and `nwdaf-anlf` Deployments over the demo base, ClusterIP only; `scenario.sh` for `k8s/demo-check.sh`. |

### Toggles

Follows the #458 convention. Both gates are off by default.

- **Cargo:** nwdafd feature `demo-d441e-model-retrain`.
- **Config** (both nwdafd instances, config plus restart):

  ```yaml
  nwdaf:
    role: anlf                  # the AnLF instance; mtlf on the other
    anlf:
      mtlf_uri: http://nwdaf-mtlf:7777   # optional; default: NRF discovery
      accuracy_threshold: 70             # optional, percent
  demo:
    enabled: true
    d441e-model-retrain:
      enabled: true
      mode: retrain             # or no-retrain
  ```

| Mode | Behaviour |
|---|---|
| `retrain` | The MTLF refits on an accuracy drop and provisions v2. |
| `no-retrain` | The MTLF never refits. Accuracy stays below the threshold. |

The AnLF behaves the same in both modes.

**Two instances are needed.** The AnLF client runs only on `role: anlf`. A
single `role: both` NWDAF runs no client: TS 23.288 §6.2E.3.1 leaves the
co-located case implementation-dependent.

When the demo is active, nwdafd logs `WARN DEMO MODE ACTIVE: d441e-model-retrain=<mode>`,
prefixes its lines with `[demo:d441e-model-retrain]`, and exports
`nextg_demo_enabled{demo="d441e-model-retrain",mode="<mode>"} 1`. An AnLF also
exports these metrics:
- `nwdaf_anlf_client_active`;
- `nwdaf_anlf_mtlf_subscribed`;
- `nwdaf_anlf_models_total{outcome="verified"|"refused"}`.

**Startup failures:**
- `mode: bogus` fails with
  `UnknownMode { demo: "d441e-model-retrain", mode: "bogus", known: ["retrain", "no-retrain"] }`;
- the key in a build without the feature fails with `UnknownKey`;
- an unknown `nwdaf.role` fails;
- a malformed `nwdaf.anlf` block fails.

### Log chain (AnLF)

```text
[demo:d441e-model-retrain] model-retrain: on (mode=retrain, role=anlf; AnLF client to ..., threshold=0.70)
[demo:d441e-model-retrain] subscribed to MTLF <id> threshold=0.70
[demo:d441e-model-retrain] model v1 verified sha256=<sha256> from <mLModelUrl>
[demo:d441e-model-retrain] model refused (sha256 mismatch) from <mLModelUrl>: ...; keeping v1
```

On the MTLF:
- `MLModelProvision: model vN published sha256=...`;
- `MLModelProvision: model vN provisioned sha256=... sub=...`.

## How to run it

### Tests

```bash
cd nextgcore/src
cargo test -p nextgcore-nwdafd --features demo-d441e-model-retrain   # AnLF client over HTTP against nwdafd's MTLF handlers
cargo test -p nextgcore-nwdafd                                        # demo-off and ungated MTLF behaviour
```

### Compose

nextgcore's `docker/rust/docker-compose.demo-d441e.yml` is standalone: an NRF
and the two NWDAFs, nothing published, an `internal` network. Build nrfd and
nwdafd in one cargo call, `cargo build --release -p nextgcore-nrfd -p
nextgcore-nwdafd --features nextgcore-nwdafd/demo-d441e-model-retrain` (the
feature is nwdafd's; nrfd has none), and package each with `Dockerfile.nf`.
Then, from `nextgcore/docker/rust`:

```bash
export D441E_NWDAF_IMAGE=<nwdaf image> D441E_NRF_IMAGE=<nrf image>
configs/demo-d441e/check.sh retrain        # the chain to model v2 verified
configs/demo-d441e/check.sh no-retrain     # the drop, no retrain, no v2
configs/demo-d441e/check.sh off-absent     # also off-disabled, off-key-only
configs/demo-d441e/check.sh bogus-mode     # UnknownMode fails start-up
configs/demo-d441e/reverts.sh              # flipped v2 byte; FeedbackRegistry feed patched out
```

`reverts.sh` rebuilds nwdafd under each patch. It needs `D441E_CORE_IMAGE`, the
base image `Dockerfile.nf` extends (default `nextgcore-core:latest`, built from
`Dockerfile.core`), and a checkout whose tracked files match HEAD. It refuses a
dirty tree, and it reverses each patch right after that patch's build.

A seeded arm waits up to 10 minutes. The series is a function of unix time,
and the NWDAFs are started 25-60 s into a 600 s cycle. The AnLF reports as soon
as the MTLF subscribes, over the newest 20 points, so a start just after a
regime change would request a retrain before any calm report. nextgcore CI runs
every arm nightly (job `demo-d441e`).

### EKS

```bash
flock /tmp/eks-nextg-demo.lock k8s/demo-check.sh d441e-model-retrain retrain
flock /tmp/eks-nextg-demo.lock k8s/demo-check.sh d441e-model-retrain no-retrain
flock /tmp/eks-nextg-demo.lock k8s/demo-check.sh off off
```

The overlay `deploy/eks/overlays/demo-d441e-model-retrain/` adds the
`nwdaf-mtlf` and `nwdaf-anlf` Deployments to the demo base. Its `scenario.sh`
scales both to 0, then back to 1 inside the start window, and asserts the same
chain as the compose check (`chain.py`), by kubelet timestamp.

## Results

Captured on EKS devtest1 on 2026-11-13 with `k8s/demo-check.sh`, one run per
arm, each holding `/tmp/eks-nextg-demo.lock`, in namespace `nextg-demo`, which
the run created and deleted. Images: nextgcore main `42626daa65fa`
(`nextg/*:demo-d441e-model-retrain-42626daa65fa`) and nextgsim main
`e5a6875dc878` (`nextg/{gnb,ue}:demo-d441e-model-retrain-e5a6875dc878`).
Every line below is copied from the run's artefacts under
`/tmp/eks-val/demos/<key>/<mode>/<UTC>/`. The prefix is the kubelet timestamp;
nwdafd's own `[time LEVEL module]` header is cut.

| Arm | Run | Verdict | What it showed |
|---|---|---|---|
| `retrain` | `20261113T152212Z` | PASS (deploy, base e2e rc 0, scenario rc 0) | v1 at 0.98 on the calm part, `0.64 < 0.70: retrain requested` after the change, v2 verified and reporting 0.99 |
| `no-retrain` | `20261113T153644Z` | PASS (deploy, base e2e rc 0, scenario rc 0) | the same drop to 0.64, `mode no-retrain, no retrain`, then 5 reports with max 0.64 and no v2 |
| demo-off (`off off`) | `20261113T155742Z` | PASS (deploy, base e2e rc 0) | `demo_banner_lines=0 demo_prefix_lines=0 nextg_demo_enabled_series=0` |

### `retrain`: the ordered chain (AnLF and MTLF, by kubelet timestamp)

```text
2026-11-13T15:30:28.679298228Z DEMO MODE ACTIVE: d441e-model-retrain=retrain
2026-11-13T15:30:30.704629039Z [demo:d441e-model-retrain] subscribed to MTLF d441e000-0468-4000-8000-00000000f001 threshold=0.70
2026-11-13T15:30:58.842524344Z MLModelProvision: model v1 provisioned sha256=335f9636618d6577c50e2538c294a6716ac20169af37929050f016407495fa23 sub=mlsub-2a287fa1-580e-4391-9970-fa3bf943708c
2026-11-13T15:30:58.846305025Z [demo:d441e-model-retrain] model v1 verified sha256=335f9636618d6577c50e2538c294a6716ac20169af37929050f016407495fa23 from http://10.10.17.155:7777/nnwdaf-mlmodelprovision/v1/models/NF_LOAD/335f9636618d6577c50e2538c294a6716ac20169af37929050f016407495fa23
2026-11-13T15:30:58.848500447Z [demo:d441e-model-retrain] MLModelMonitor registered at MTLF 10.10.17.155:7777 modelId=56485635056013 registrationId=reg-1-cb13635a
2026-11-13T15:30:58.896014065Z [demo:d441e-model-retrain] MLModelMonitor subscribed at AnLF d441e000-0468-4000-8000-00000000a001 sub=mmsub-1-edbe17d4 modelId=56485635056013 threshold=0.70
2026-11-13T15:30:59.682712699Z [demo:d441e-model-retrain] accuracy 0.98 (n=20) reported for modelId=56485635056013 sub=mmsub-1-edbe17d4
2026-11-13T15:32:00.790120153Z [demo:d441e-model-retrain] seeded series: regime alternating from step 1794583920 (scripted regime change)
2026-11-13T15:32:29.682158278Z [demo:d441e-model-retrain] accuracy 0.64 reported for modelId=56485635056013 (reg-1-cb13635a); rolling 0.87 over 3
2026-11-13T15:32:59.681874788Z [demo:d441e-model-retrain] accuracy 0.63 reported for modelId=56485635056013 (reg-1-cb13635a); rolling 0.75 over 3
2026-11-13T15:33:29.681780330Z [demo:d441e-model-retrain] accuracy 0.64 reported for modelId=56485635056013 (reg-1-cb13635a); rolling 0.64 over 3
2026-11-13T15:33:29.681796583Z [demo:d441e-model-retrain] accuracy 0.64 < 0.70: retrain requested (modelId=56485635056013)
2026-11-13T15:33:33.790296806Z [demo:d441e-model-retrain] retrain: model v2 published sha256=14a62b901d64df4f52856e1c140b166205a984d225b7ae3ec6a848c137917038 modelId=22703927991652 (replaces v1)
2026-11-13T15:33:58.852653026Z MLModelProvision: model v2 provisioned sha256=14a62b901d64df4f52856e1c140b166205a984d225b7ae3ec6a848c137917038 sub=mlsub-2a287fa1-580e-4391-9970-fa3bf943708c
2026-11-13T15:33:58.856311241Z [demo:d441e-model-retrain] model v2 verified sha256=14a62b901d64df4f52856e1c140b166205a984d225b7ae3ec6a848c137917038 from http://10.10.17.155:7777/nnwdaf-mlmodelprovision/v1/models/NF_LOAD/14a62b901d64df4f52856e1c140b166205a984d225b7ae3ec6a848c137917038
2026-11-13T15:33:58.908576790Z [demo:d441e-model-retrain] MLModelMonitor registered at MTLF 10.10.17.155:7777 modelId=22703927991652 registrationId=reg-2-5ef84d41
2026-11-13T15:33:59.682467201Z [demo:d441e-model-retrain] accuracy 0.99 (n=20) reported for modelId=22703927991652 sub=mmsub-2-f9ef63d5
```

From the MTLF's `retrain: AR(5) ...` line and its `/metrics`:

```text
held-out MAE 0.009 refit, 0.358 current model (no-retrain), 0.593 persistence
nwdaf_model_refits_total 1
nwdaf_model_refit_holdout_mae{model="refit"} 0.009197153038983899
nwdaf_model_refit_holdout_mae{model="current"} 0.35752852437852917
nwdaf_model_refit_holdout_mae{model="persistence"} 0.593100405762858
```

After v2, the scenario fetched both artefact URLs from inside the AnLF. The v1
URL still serves v1 (`scenario/artefacts.txt`):

```text
v1 http://10.10.17.155:7777/nnwdaf-mlmodelprovision/v1/models/NF_LOAD/335f9636618d6577c50e2538c294a6716ac20169af37929050f016407495fa23 sha256=335f9636618d6577c50e2538c294a6716ac20169af37929050f016407495fa23
v2 http://10.10.17.155:7777/nnwdaf-mlmodelprovision/v1/models/NF_LOAD/14a62b901d64df4f52856e1c140b166205a984d225b7ae3ec6a848c137917038 sha256=14a62b901d64df4f52856e1c140b166205a984d225b7ae3ec6a848c137917038
```

Both NWDAFs' scrapes carry `nextg_demo_enabled{demo="d441e-model-retrain",mode="retrain"} 1`.

### `no-retrain`: the same seeded run, no refit

```text
2026-11-13T15:43:28.415755725Z [demo:d441e-model-retrain] accuracy 0.64 < 0.70: mode no-retrain, no retrain (modelId=56485635056013)
2026-11-13T15:43:58.415637307Z [demo:d441e-model-retrain] accuracy 0.63 (n=20) reported for modelId=56485635056013 sub=mmsub-1-9c95b36c
2026-11-13T15:44:28.415956034Z [demo:d441e-model-retrain] accuracy 0.64 (n=20) reported for modelId=56485635056013 sub=mmsub-1-9c95b36c
2026-11-13T15:44:58.417185140Z [demo:d441e-model-retrain] accuracy 0.64 (n=20) reported for modelId=56485635056013 sub=mmsub-1-9c95b36c
2026-11-13T15:45:28.415758560Z [demo:d441e-model-retrain] accuracy 0.64 (n=20) reported for modelId=56485635056013 sub=mmsub-1-9c95b36c
```

Before the change, v1 reported 0.99 on the calm part. There was no
`retrain requested`, no `retrain:` line and no v2; `nwdaf_model_refits_total 0`
on the MTLF; and `nextg_demo_enabled{demo="d441e-model-retrain",mode="no-retrain"} 1`
on both NWDAFs.

### Baseline against the mode

On the same series, after the change:
- v1 kept in use (`no-retrain`) scores 0.63-0.64;
- the refit v2 (`retrain`) scores 0.99;
- on the held-out rows, the MAE is 0.009 for the refit, 0.358 for the current
  model and 0.593 for persistence.

The MAE is reported, not asserted.

### Compose and the revert arms

nextgcore CI's `demo-d441e` job (run
[43403156885](https://github.com/NextgCoreLab/nextgcore/actions/runs/43403156885),
artefact `demo-d441e`) passed every arm on the compose stack. It ran on PR
#1495's head `81e63125`, whose tree is identical to main `42626daa` (the same
git tree object, so the same under `docker/`, `deploy/`, `k8s/`, `.github/` and
nwdafd):
- bogus-mode: both NWDAFs exit 1 with
  `UnknownMode { demo: "d441e-model-retrain", mode: "bogus", known: ["retrain", "no-retrain"] }`;
- off-absent, off-disabled and off-key-only: the MLModelMonitor routes answer
  404, and there is no demo line and no `nextg_demo_enabled`;
- retrain and no-retrain;
- both revert arms:
  - with one byte of v2 flipped, the check fails at S12, with the AnLF logging
    `model refused (sha256 mismatch) from …/5ee6c246…; keeping v1`;
  - with the `FeedbackRegistry` feed patched out, the check fails at S8: no
    `retrain requested`, while the AnLF reports `accuracy 0.64 (n=20)`.

## Honesty labels and limitations

- The model is OLS in a single `ai.onnx.ml` LinearRegressor. "Learned" means
  refit on recent data.
- The manifest comes from the same MTLF as the bytes. The sha256 check proves
  the transfer, not the provenance: TS 33.501 Annex X.10 is not implemented.
- Version labels (v1, v2) restart with the MTLF process. The sha256 is the
  identity.
- Accuracy is `1 - mean |error|` of a one-step backtest over the newest 20
  points, rolled over 3 reports (the `FeedbackCollector` definition). It is not
  the TS 23.288 §5C.1 methods.
- The series is synthetic and its regime change is scripted and disclosed:
  both NWDAFs log `seeded series: regime alternating ... (scripted regime
  change)`. The MTLF retrains on its own copy of the series, not on data
  fetched from the AnLF (§6.2E.3.3 step 9 allows either).
- The held-out MAE is reported against the current model and persistence. It
  is not asserted to be better.
- The two NWDAFs are the same binary in different roles. `role: both` runs no
  AnLF client, so the demo always uses two instances.
- **Not under SBI OAuth2 enforcement yet.** With `NEXTGCORE_SBI_OAUTH2_REQUIRE=1`
  (or `nwdaf.sbi.oauth2.require: true`) on the AnLF, the MTLF's token-less
  Notify is refused with 401, although TS 33.501 §13.4.1 says OAuth 2.0 does
  not apply to notifications. The callback authenticates by its 122-bit
  token instead. Until
  [nextgcore#1421](https://github.com/NextgCoreLab/nextgcore/issues/1421)
  lands, the chain cannot run under OAuth2 enforcement. The compose and EKS
  stacks run without it, and the ticket's OAuth2 criterion is still open.
- The AnLF's verification is nwdafd's own reimplementation of #463's checks,
  with no ONNX Runtime.
- MTLF provisioning serves one lineage (NF_LOAD) for every subscribed event.
  That gap is filed as nextgcore#1452.
