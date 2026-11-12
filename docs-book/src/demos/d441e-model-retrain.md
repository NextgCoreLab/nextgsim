# D441E: MTLF→AnLF model provisioning with an accuracy-triggered retrain

Demo key `d441e-model-retrain`, ticket
[nextgsim#468](https://github.com/NextgCoreLab/nextgsim/issues/468), program
anchor [nextgsim#458](https://github.com/NextgCoreLab/nextgsim/issues/458).
The code is in nextgcore nwdafd and lands in five PRs:

- PR 1, [nextgcore#1454](https://github.com/NextgCoreLab/nextgcore/pull/1454) (merged): the
  MTLF's versioned model store;
- PR 2, [nextgcore#1464](https://github.com/NextgCoreLab/nextgcore/pull/1464): the AnLF client and its verification;
- PR 3: MLModelMonitor;
- PR 4: the retrain;
- PR 5: the compose and EKS overlays and the check script.

> **Status:** in progress. This page describes the whole demo. The parts that
> PRs 3-5 add are marked *(planned)*. No result has been captured yet.

## Purpose

To show the TS 23.288 §6.2A/§6.2E split working between two real NWDAF
instances. An MTLF provisions a verified model to an AnLF, the AnLF reports
accuracy, and a measured drop triggers retraining and re-provisioning of v2.
The whole cycle is observed in CI.

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
  Note over AnLF,MTLF: (planned, PR 3) Nnwdaf_MLModelMonitor Register / Subscribe / Notify (accuracy)
  Note over MTLF: (planned, PR 4) rolling accuracy < 0.70: retrain, publish v2
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
| AnLF client | nwdafd `model_retrain` (PR 2, gated) | On `role: anlf` with the demo on: finds the MTLF (`nwdaf.anlf.mtlf_uri`, else NRF discovery), subscribes, and takes the Notify on `/nnwdaf-callback/v1/mtlf-model-notify/{token}` on the existing SBI listener. It downloads the model and manifest, verifies them with #463's checks and loads the model as its NF_LOAD predictor. A refused model leaves the current one in place. |
| MLModelMonitor *(planned, PR 3)* | nwdafd | Register at the MTLF, Subscribe at the AnLF, accuracy Notify. |
| Retrain *(planned, PR 4)* | nwdafd MTLF | A least-squares refit on the recent window when rolling accuracy drops below the threshold. |

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
| `retrain` | The MTLF refits on an accuracy drop and provisions v2 *(planned, PR 4)*. |
| `no-retrain` | The MTLF never refits. Accuracy stays below the threshold *(planned, PR 4)*. |

The AnLF behaves the same in both modes.

When the demo is active, nwdafd logs `WARN DEMO MODE ACTIVE: d441e-model-retrain=<mode>`,
prefixes its lines with `[demo:d441e-model-retrain]`, and exports
`nextg_demo_enabled{demo="d441e-model-retrain",mode="<mode>"} 1`. An AnLF also
exports these metrics:
- `nwdaf_anlf_client_active`;
- `nwdaf_anlf_mtlf_subscribed`;
- `nwdaf_anlf_models_total{outcome="verified"|"refused"}`.

**Startup failures:**
- `mode: bogus` fails with
  `UnknownMode { demo: "d441e-model-retrain", mode: "bogus" }`;
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

### Compose and EKS *(planned, PR 5)*

```bash
k8s/demo-check.sh d441e-model-retrain retrain
k8s/demo-check.sh d441e-model-retrain no-retrain
k8s/demo-check.sh off off
```

## Results

**Not yet run.** This page will hold the captured output once PR 5's overlays
exist and the EKS arms (demo-off, `retrain`, `no-retrain`) have run on
devtest1. Nothing here is a result.

## Honesty labels and limitations

- The model is OLS in a single `ai.onnx.ml` LinearRegressor. "Learned" means
  refit on recent data.
- The manifest comes from the same MTLF as the bytes. The sha256 check proves
  the transfer, not the provenance: TS 33.501 Annex X.10 is not implemented.
- Version labels (v1, v2) restart with the MTLF process. The sha256 is the
  identity.
- Accuracy is `1 - mean |error|` over a window (the `FeedbackCollector`
  definition), not the TS 23.288 §5C.1 methods *(planned, PR 3)*.
- The regime change is scripted and disclosed *(planned, PR 4)*.
- The two NWDAFs are the same binary in different roles.
- MTLF provisioning serves one lineage (NF_LOAD) for every subscribed event.
  That gap is filed as nextgcore#1452.
