# D445A: network-assisted federated learning over real PDU sessions

Ticket: [nextgsim#480](https://github.com/NextgCoreLab/nextgsim/issues/480), program anchor [#458](https://github.com/NextgCoreLab/nextgsim/issues/458). Demo key `d445a-fl-t8`.

## Purpose

Show 5G-Advanced reviewers that an application-layer FL server (an AF) trains over real UE PDU sessions, that a NEF T8 Loss-of-Connectivity event removes a departed member before the next round, and that learning is proven against honest baselines. That is TS 23.501 §5.46.

- **Success signal:** after `ue3` deregisters, the round that follows logs `participants=2`, preceded by `member_selection excluded=<gpsi> reason=nef_event`. A strictly ordered amfd → nefd → AF chain comes before that line. The AF's held-out MSE beats local-only.
- **Not the point:** this is not a gNB aggregator, not split training, and not a privacy proof.

## What holds on every seed, and what holds on seed 480 only

The demo pins seed 480. The library was measured on 23 seeds (480-499, 1, 7 and 42; `nextgsim-fl` `sim` module, nextgsim#528). These claims hold on **every** seed measured:

- every round's held-out MSE is finite, and so are both baselines;
- after 10 rounds the federation beats the mean local-only model (≤ 0.8× local-only on 23 of 23);
- in the poisoned-UE arm, Krum scores UE 005 worst in every round and never selects it (23 of 23).

These claims are **seed-480 results**, from one run each, not properties of the setup:

- round 10 ≤ 0.5× round 1 (19 of 23 seeds);
- round 10 ≤ 1.2× pooled (3 of 23: 480, 495, 1);
- median and trimmed mean each within 1.1× of their own run without the attacker (both together on 5 of 23: 480, 481, 489, 498, 499).

On seed 480 (library run, x86_64 glibc, debug and release identical) the federation reaches 0.184× round 1, 0.466× local-only and 1.069× pooled. In the poisoned arm, median is 0.937× and trimmed mean (0.4) 1.042× their clean runs, and FedAvg diverges (about 1.9e6×).

Seed table (lr 0.15, gradient-norm clip 1.0; each ratio is round-10 MSE over the baseline; the robust columns are poisoned over clean for the same rule):

| seed | r10/r1 (≤0.5) | r10/local (≤0.8) | r10/pooled (≤1.2) | median (≤1.1) | trimmed (≤1.1) | Krum worst/selected |
|---|---|---|---|---|---|---|
| 480 | 0.184 | 0.466 | 1.069 | 0.937 | 1.042 | 10/0 |
| 481 | 0.017 | 0.069 | 2.508 | 0.903 | 0.945 | 10/0 |
| 482 | 0.050 | 0.344 | 2.670 | 2.964 | 1.954 | 10/0 |
| 483 | 0.106 | 0.549 | 2.333 | 2.038 | 1.828 | 10/0 |
| 484 | 0.618 | 0.530 | 1.740 | 1.050 | 1.103 | 10/0 |
| 485 | 0.308 | 0.525 | 2.334 | 1.461 | 1.445 | 10/0 |
| 486 | 0.527 | 0.713 | 3.011 | 1.189 | 1.927 | 10/0 |
| 487 | 0.171 | 0.258 | 1.256 | 0.991 | 1.248 | 10/0 |
| 488 | 0.581 | 0.271 | 3.704 | 1.365 | 2.291 | 10/0 |
| 489 | 0.275 | 0.516 | 1.337 | 1.068 | 1.042 | 10/0 |
| 490 | 0.330 | 0.279 | 2.279 | 1.099 | 1.316 | 10/0 |
| 491 | 0.248 | 0.317 | 4.158 | 1.170 | 1.489 | 10/0 |
| 492 | 0.234 | 0.533 | 2.167 | 1.877 | 2.260 | 10/0 |
| 493 | 0.261 | 0.114 | 2.692 | 1.753 | 1.918 | 10/0 |
| 494 | 0.088 | 0.522 | 1.670 | 1.870 | 1.920 | 10/0 |
| 495 | 0.235 | 0.568 | 1.153 | 1.252 | 1.320 | 10/0 |
| 496 | 0.431 | 0.648 | 4.289 | 1.836 | 1.913 | 10/0 |
| 497 | 0.244 | 0.312 | 2.154 | 2.561 | 4.232 | 10/0 |
| 498 | 0.457 | 0.582 | 2.106 | 0.912 | 0.980 | 10/0 |
| 499 | 0.133 | 0.420 | 2.915 | 0.579 | 0.825 | 10/0 |
| 1 | 0.028 | 0.614 | 1.113 | 1.760 | 2.030 | 10/0 |
| 7 | 0.855 | 0.644 | 7.499 | 2.196 | 1.816 | 10/0 |
| 42 | 0.132 | 0.367 | 2.243 | 1.712 | 2.269 | 10/0 |

"The clean run" is read per rule: each robust rule against its own run without the attacker. Against clean FedAvg, median and trimmed mean sit at about 1.11× and 1.08× on seed 480, which is their cost without any attacker.

Reruns are bit-identical **per platform image**: libm's `tanh`, `ln` and `cos` differ across glibc, musl and macOS, and across x86_64 and aarch64, so an x86_64 AF and an arm64 UE are not expected to match bit for bit.

## How it works

```mermaid
sequenceDiagram
  participant UE as UE FL client (nr-ue)
  participant AF as nr-fl-server (AF)
  participant NEF as nefd
  participant BSF as bsfd
  participant UDM as udmd
  participant AMF as amfd
  UE->>AF: Hello over the PDU session (uesimtunN, GTP-U, N6 without NAT)
  AF->>NEF: UE ID retrieval, TS 29.522 UeIdReq (afId, ueIpAddr)
  NEF->>BSF: Nbsf_Management_Discovery (ipv4Addr) -> SUPI
  NEF->>UDM: Nudm_SDM GetSupiOrGpsi -> GPSI
  NEF->>AF: UeIdInfo externalId (the AF-specific External Identifier)
  AF->>NEF: T8 Monitoring Event subscription, LOSS_OF_CONNECTIVITY, externalId
  NEF->>UDM: Nudm_EE_Subscribe
  UDM->>AMF: Namf_EventExposure_Subscribe (TS 23.502 §4.15.3.2.3 step 3a)
  AF->>UE: GlobalModel (round K)
  UE->>AF: Update (delta, samples, loss)
  UE->>AMF: Deregistration Request (ue3)
  AMF->>NEF: Namf_EventExposure_Notify (LOSS_OF_CONNECTIVITY, DEREGISTERED)
  NEF->>AF: T8 MonitoringNotification
  AF->>AF: exclude ue3; round K+1 at participants=2
```

Components:

| Piece | Where | Gate |
|---|---|---|
| Seeded task, MLP, frames, aggregation rules | `nextgsim-fl` (`seeded`, `synthetic`, `mlp`, `frame`, `sim`) | library, no runtime effect |
| `nr-fl-server` AF: FL listener, rounds, scorecard | `nextgsim-fl/src/d445a/server.rs`, `src/bin/nr_fl_server.rs` | cargo `demo-d445a-fl-t8`; refuses to start unless `demo.d445a-fl-t8` is enabled |
| NEF client and T8 callback listener (h2c) | `nextgsim-fl/src/d445a/nef.rs` | same |
| UE FL client, `fl-train`, `fl-status` | `nextgsim-fl/src/d445a/client.rs`, `nextgsim-ue/src/fl_participant/d445a.rs` | cargo `demo-d445a-fl-t8` on `nextgsim-ue` + `demo:` block + `federated_learning_enabled` + `fl:` block |
| Revert knobs `--no-t8`, `--freeze-global`, `fl.leak_raw` | the same files | cargo `revert-hooks`; refused without it |
| UE ID retrieval, compose and EKS overlays, `demo-check.sh` | nextgcore (plan item 7) | `demo-d445a-fl-t8` + block in nefd |

The gNB has no FL task: its aggregator was removed (nextgsim#480). Federated learning is an application on the N6 side.

### The AF

- **Join.** A UE says `Hello { label: ueN }`. The AF asks the NEF who is behind the TCP source address (TS 29.522 UE ID retrieval). The answer's `externalId` becomes the member's GPSI `extid-<externalId>`. Only then does the AF subscribe to T8 `LOSS_OF_CONNECTIVITY` for it. A peer the NEF cannot identify is refused (`identity_unresolved`). That is what ties a member to a real PDU session. A label that comes back with a different GPSI, or a GPSI already bound to another label, is refused (`identity_mismatch`). The refusal comes before any T8 subscription is made for it, so one UE cannot hold two weights or take over another member's label. There is no per-source join rate limit: every Hello still costs one UE ID call, bounded by the 16-connection cap and the 10 s Hello timeout.
- **Rounds.** Each round selects the joined members that are not excluded, sends each the global model, and waits for their updates until all are in or the straggler deadline T (`round_deadline_ms`, default 30 s) passes (`straggler_timeout`). The AF aggregates with the library's rule, applies the result and scores the model on the held-out set.
- **Exclusion.** A T8 LoC report on one of the AF's own subscriptions excludes the member at once, and the AF stops waiting for it: `member_selection excluded=<gpsi> member=ue3 reason=nef_event lossOfConnectReason=…`. A TCP EOF or RST is only logged (`transport_closed`). After a deregistration or a power cycle there is no user plane left to carry a FIN, as with a real handset that loses radio.
- **Readmission.** The same identity joining again is `member_selection readmitted=<gpsi>` from the next round. Its old subscription is deleted and a new one made.
- **Holds** make the check deterministic. `holds: [{before_round: 4, until: member-excluded}, {before_round: 7, until: members-full}]` pauses before round 4 until a T8 exclusion arrives, and before round 7 until everyone is back (each with a timeout).
- **Scoring.** After each arm: `scorecard rule=… r1_mse last_mse local_only_mse pooled_mse last/r1 last/local last/pooled oracle_bit_identical`. The AF computes the baselines itself from the pinned synthetic task. `oracle_bit_identical` replays the run in process with the membership each round actually had. In `poisoned-ue` there is also `robustness rule=… poisoned_last_mse clean_last_mse ratio`, plus a `member_selection … rule=krum excluded=<gpsi> reason=krum_worst` line per Krum round.

### Trust and exposure

- `Update.samples` is client-supplied and is the FedAvg weight. The AF caps it at `samples`, the per-member count of its task (`samples_capped`), so a `u32::MAX` claim weighs the same as an honest one.
- An update's delta is client-supplied too. `max_update_norm` is a **crash guard, not a poisoning defence**. An attacker below the cap (such as UE 005's ×10 deltas, about 90) is aggregated as usual, and robustness against it is the rules' job (median, trimmed mean, Krum). A delta whose L2 norm exceeds `max_update_norm` (default 1000; the largest seed-480 update, the poisoned FedAvg arm's, is about 90) is refused and not aggregated (`update_refused … reason=norm`). A round whose aggregate would make any weight non-finite is refused and the previous global carries over (`refused="non-finite"`), in the AF and in the library's in-process run alike. So a member sending `f32::MAX` deltas cannot crash the AF or end an arm early.
- Frames are the strict `NFL1` codec: exactly 261 finite floats, a short plain label, at most 1 MiB. A refused frame closes the connection (`frame_refused`). Every received byte is scanned for the task's canary rows; the run ends with `canary_rows=<n>`.
- Both AF listeners bind 127.0.0.1 by default, and the AF publishes no port (ClusterIP only). `listen` (FL) binds the `fl-dn` address the UEs reach through the UPF. `t8_listen` belongs on the network nefd reaches the AF on (the SBI or pod network), never on `fl-dn`. The AF never advertises a wildcard (`t8_notify_base` must name a host).
- The T8 callback listener answers only `POST /fl-af/v1/t8-notify/<id>` for ids the AF minted (128 random bits, given only to the NEF), with a 64 KiB body cap. Any other id gets 404. A notification whose `subscription` link is not the AF's is ignored. It allows at most 16 streams per connection and a 16 KiB header list, and closes a connection that has had nothing in flight for 30 s. Eviction rules:
  - One source address holds at most 8 connections. Past that, it evicts only its own oldest idle connection, never another source's.
  - At 64 connections in all, an eviction takes from the source holding the most, and only if that source holds at least two more than the newcomer's source. A peer holding one connection is never evicted by a newcomer.
  - A connection with a request in flight is never evicted. A connection still opening (accept, h2 handshake, first request, at most 2 s) is not evicted by its own source.

  So one flooding or churning host, or a flood spread over many addresses, cannot evict nefd's connection. Deterministic tests of the eviction rules prove this, including their boundaries. Two churn smoke tests (200 notifications under a flood, with no loss) run separately and serially.

  Known limits:
  - **nefd's address must not be shared.** Slots are counted per source address, so nefd must reach `t8_listen` on an address that no untrusted peer shares: pod-to-pod, with no SNAT between nefd and the AF, no hostNetwork, no NodePort or load-balancer path, and not the default 127.0.0.1 alongside other local processes. A peer that shares nefd's address shares its 8 slots, and could keep them all opening.
  - **Many hosts can still refuse nefd.** With `k` notifications in flight at once, nefd's k-th connection is refused only when every slot is held by a source with at most `k` connections. That takes about 64/k hosts: more than 64 for one notification at a time, about 32 for two simultaneous loss-of-connectivity reports.
  - **IPv6 is not covered.** On IPv6 one host with a /64 counts as many sources. The listener is IPv4 (pod network or loopback) here.
  - **A raw SYN flood** is below this layer and is the kernel's to absorb (SYN cookies).
- An evicted connection had nothing in flight as the registry last saw it, but a request may already be on the wire. It therefore drains: GOAWAY, then the streams the peer opens in the first 250 ms are answered. Later ones are refused (`REFUSED_STREAM`), and at 2 s from the start of the drain the connection is closed whatever is in flight. No stream extends these deadlines. A connection closing after 30 s idle drains the same way and is never picked as an eviction victim.
  - **The socket bound.** Draining connections are out of the admission counts, so they never cause a refusal. They are capped on their own at 64; past that, the one draining longest is closed at once. The listener therefore holds at most 64 live plus 64 draining connections, each draining for at most 2 s. A connection evicted before its h2 handshake completes is closed at once.
- A refused connection is logged at debug level. A warn summary with the count goes out at most every 10 s, and the last window of a burst is flushed on a timer.
- With `nef.oauth2` every NEF call carries an NRF token for an `AF`-typed consumer, or is not sent. The scopes `nnef-ueid` and `nnef-eventexposure` are nextgcore's NRF OAuth2 policy, not 3GPP surface: the 3GPP OpenAPI defines no scopes for these northbound APIs, and TS 23.222 / TS 33.122 put them behind CAPIF.

Modes (`demo.d445a-fl-t8.mode`, default `t8-exclusion` in the overlay):

- `t8-exclusion`: 3 UEs. `ue3` deregisters, the NEF exclusion follows, then readmission. Rule FedAvg.
- `poisoned-ue`: 5 UEs. UE 005 sends ×10 sign-flipped deltas. The AF runs four arms from the same seed (FedAvg, median, trimmed mean 0.4, Krum f=1) and scores each against its own clean run.

Krum needs n > 2f + 2 members. With f = 1 that is 5 or more, so in the 3-member `t8-exclusion` shape every Krum round is refused and names nobody. The exclusion line there comes from T8 only.

Switching: `nr-cli <ue-005> demo d445a-fl-t8 set mode=<mode>` flips poisoning from the next round (audited; `mode=bogus` is rejected and the mode stays). The AF and nefd switch by config plus restart. `nr-cli <ue> fl-status` shows the member, state, sessions, rounds trained and the last loss. `nr-cli <ue> fl-train` starts a client configured with `autostart: false`.

## How to run it

UE config, on top of a 6G-capable UE config:

```yaml
federated_learning_enabled: true
fl:
  server: 10.46.0.10:7480   # the AF on fl-dn, reached through the PDU session
  device: uesimtun0         # bind the session's TUN
  member: 3
  clients: 3                # 5 in poisoned-ue
demo:
  enabled: true
  d445a-fl-t8: { enabled: true, mode: t8-exclusion }
```

Build the UE with `--features demo-d445a-fl-t8` and the AF with `cargo build -p nextgsim-fl --features demo-d445a-fl-t8 --bin nr-fl-server`. The revert arms need `--features revert-hooks`.

The compose overlay (a `nef` service, the `internal: true` `fl-dn` network, upfd `--nat-exclude`) and the EKS overlay `deploy/eks/overlays/demo-d445a-fl-t8/` are in nextgcore (plan item 7). Then:

```bash
flock /tmp/eks-nextg-demo.lock k8s/demo-check.sh d445a-fl-t8 t8-exclusion
flock /tmp/eks-nextg-demo.lock k8s/demo-check.sh d445a-fl-t8 poisoned-ue
flock /tmp/eks-nextg-demo.lock k8s/demo-check.sh d445a-fl-t8 off
```

In-process arms (no core needed): `cargo test -p nextgsim-fl --features revert-hooks d445a` runs the AF against a stub nefd over h2c, with real TCP clients on distinct loopback source addresses. It covers exclusion and readmission, a mid-round exclusion, `--no-t8`, `--freeze-global`, `leak-raw`, the poisoned arm, the samples cap and an unidentifiable peer. Code revert arms: `scripts/demo/d445a-reverts.sh`.

## Results

All twelve arms pass on devtest1 EKS (us-west-2, cluster `devtest1`, namespace `nextg-demo`), driven from the DRAFT nextgcore PR [#1492](https://github.com/NextgCoreLab/nextgcore/pull/1492). Images
`nextg/{nrf,amf,smf,upf,ausf,udm,udr,pcf,nssf,bsf,nef}:demo-d445a-fl-t8-d477985d6d44`
(nextgcore `d477985d6d44` from the build tree) and `nextg/{gnb,ue}:demo-d445a-fl-t8-96a2933668ab` (nextgsim `96a2933668ab`).

The run surfaced six harness defects, each fixed on the PR branch and verified by rerun (nextgcore commits
`d77aca49` pipe-into-grep under pipefail, `7d94bc46` the delivery lines are not ordered against round 4,
`e6f4a81f` probe cycle 1 matches a first join, `00d27996` the demo rollout waits for demo-added NFs, `cec73d25` `auth[@]` under `set -u`, `1827d9cd` the nefd bogus gate line stays on one line). The off, main, poisoned, no-t8, freeze-global, leak-raw, nat-on, nefd-down and udm-fallback arms all passed on their first attempt.

| arm | artefacts (`/tmp/eks-val/demos/...`) | VERDICT | evidence |
|---|---|---|---|
| off | `off/off/20261113T160241Z` | PASS | base e2e UL 0->2 pkts, DL 0->2 pkts; no `nextg_demo_enabled` series |
| main | `d445a-main/d445a-fl-t8/t8-exclusion/20261113T164301Z` | PASS | chain: udmd forwarded x3, from the Accept amfd delivered +1.9 ms, nefd T8 sub=d248427d-bf65-4353-bb35-19b3804c07e2 +3.7 ms, AF excluded extid-ue003@fl.nextg (DEREGISTERED) +3.4 ms, round 4 participants=2 +3.4 ms, dur_ms=3 (< 9000); scorecard rounds=10 last/r1=0.184 last/local=0.465 last/pooled=1.069 oracle_bit_identical=true; readmission rounds 7-10 participants=[3,3,3,3]; no_fin 7 segments 0 FIN/RST from the UE before the AF close; ul_delta per PDU session (0x1 +13094 >= AF 12636; 0x2 +13094 >= AF 12636; 0x3 released +3592 >= AF 3592 at the mid scrape; 0x4 +6368 >= AF 5910); AF uid 65534 CapEff/CapPrm/CapBnd 0 NoNewPrivs 1 |
| poisoned | `d445a-poisoned/d445a-fl-t8/poisoned-ue/20261113T165432Z` | PASS | sources 5 joins (.2 to .6); poisoned median 0.937 trimmed-mean 1.042 fedavg 1 855 570.625; krum 10/10 excluded extid-ue005@fl.nextg; canary 0/0; ul_delta all five sessions above AF bytes |
| probe | `d445a-probe/d445a-fl-t8/t8-exclusion/20261113T203158Z` | PASS | 10/10 cycles pass (one LoC on deregister with a reason, none within 20 s of the power cycle, readmission within 300 s) |
| gates | `d445a-gates/d445a-fl-t8/t8-exclusion/20261113T213450Z` | PASS | every gate PASS: AF nnef-ueid token 200 externalId=ue001@fl.nextg; no token 401 Bearer invalid_token; malformed token 401; NWDAF asks the NRF for nnef-ueid refused unauthorized_client; AF nnef-eventexposure-only token 403 insufficient_scope; nr-cli bogus rejected (mode stays); nr-ue bogus fails startup (rc 1); ue1 timed out on UPF:7481 and 10.46.0.10:7481, connected to 10.46.0.10:7480; AF bogus rc=1; nefd kill switch no banner, route 404 Resource not found, metrics unreachable; nefd bogus fails startup; nefd back on with the demo banner |
| no-t8 | `d445a-no-t8/d445a-fl-t8/t8-exclusion/20261113T180327Z` | PASS (revert arm holds) | chain FAILS; preconditions hold (Accept; ue3 joined t8=off; no udmd EE Subscribe, no forwarded leg); straggler_timeout round 4 member=ue3 after 30000 ms (T=30000) |
| freeze-global | `d445a-freeze-global/d445a-fl-t8/t8-exclusion/20261113T183050Z` | PASS (revert arm holds) | scorecard FAILS rounds=10 last/r1=1.0 last/local=20.317 last/pooled=46.649 oracle_bit_identical=false; precondition last/r1=1.0 within [0.95, 1.05] (the global never moved) |
| leak-raw | `d445a-leak-raw/d445a-fl-t8/t8-exclusion/20261113T185120Z` | PASS (revert arm holds) | canary FAILS canary_rows=10 refused_frames=10; precondition canary_rows > 0 and frame_refused member=ue2 |
| nat-on | `d445a-nat-on/d445a-fl-t8/t8-exclusion/20261113T191640Z` | PASS (revert arm holds) | sources FAILS (every UE identity_unresolved as 10.46.0.1, the NATed source); precondition identity_unresolved from the NATed 10.46.0.1 |
| nefd-down | `d445a-nefd-down/d445a-fl-t8/t8-exclusion/20261113T193839Z` | PASS (revert arm holds) | chain FAILS (no AF exclusion after the Accept); straggler_timeout round 4 after 30000 ms; preconditions hold (Accept, udmd x3, no AF exclusion, amfd `Event notification delivery failed` on ue3's leg sub-2a31f3b4-7b8a-4e4c-a884-53494bf13805) |
| eth0 | `d445a-eth0/d445a-fl-t8/t8-exclusion/20261113T210533Z` | PASS (revert arm holds) | sources FAILS (identity_unresolved ue2 peer=10.10.4.88); ul_delta FAILS (13 511 bytes from 10.10.4.88, no PDU session); preconditions hold (ue2 on eth0 reached the AF, identity_unresolved for ue2, ul_delta fails on the eth0 bytes, UPF member series present) |
| udm-fallback | `d445a-udm-fallback/d445a-fl-t8/t8-exclusion/20261113T201739Z` | PASS (revert arm holds) | preconditions hold (Accept, udmd could not reach the AMF, udmd forwarded no leg, no amfd 2xx LoC); INFO: ue3 was still excluded through udmd's UECM fallback (lossOfConnectReason=DEREGISTERED) |

The `main` scorecard matches the library's seed-480 line (`r10/r1=0.184`, `r10/local=0.465`, `r10/pooled=1.069`) to the printed precision, and `oracle_bit_identical=true` means the AF's seeded math on the EKS node matched the pinned oracle. The poisoned arm's `median 0.937` and `trimmed-mean 1.042` both sit at or below the 1.1 cap in the seed table, and krum selected UE 005 worst in every round, as expected.

nefd carries the `[demo:d445a-fl-t8]` UE ID retrieval block and resolves UEs by address at each T8 line (from `main/scenario/nef.log`):

```
[demo:d445a-fl-t8] ueid: on (mode=t8-exclusion; POST /3gpp-ueid/v1/retrieve, OAuth2 scope nnef-ueid required, BSF http://bsf.nextg-demo.svc.cluster.local:7777, UDM http://udm.nextg-demo.svc.cluster.local:7777, keys http://nrf.nextg-demo.svc.cluster.local:7777/nnrf-oauth2/v1/jwks)
[demo:d445a-fl-t8] ueid ipv4=10.45.0.4 afId=d445a-fl-af af=... externalId=ue003@fl.nextg
Monitoring event subscription created: id=d248427d-..., scsAsId=d445a-fl-af, type=LOSS_OF_CONNECTIVITY
[demo:d445a-fl-t8] T8 delivery sub=d248427d-... monitoringType=LOSS_OF_CONNECTIVITY lossOfConnectReason=DEREGISTERED status=204
```

The AF runs unprivileged on EKS: `main/scenario/af-status.txt` is `Uid 65534`, `CapEff/CapPrm/CapBnd 0`, `NoNewPrivs 1`. `ip netns add` worked under the pod's non-privileged capability set (`NET_ADMIN SYS_ADMIN NET_RAW CHOWN SETUID SETGID SETPCAP` plus `drop: [ALL]`); the privileged fallback was not needed.

### Harness defects surfaced by the run

- `#1492` [d77aca49](https://github.com/NextgCoreLab/nextgcore/commit/d77aca49) - three `cmd | grep -q` patterns were run under `set -uo pipefail`: a pipeline whose consumer or producer exits non-zero reported a miss even on a match. First caught in `build-images.sh`'s `nr-fl-server --no-t8` check.
- `#1492` [7d94bc46](https://github.com/NextgCoreLab/nextgcore/commit/7d94bc46) - `check_chain` gated both the amfd and nefd 2xx delivery lines on `< round 4`. The live run stamped nefd's `T8 delivery` 364 microseconds AFTER the AF's `round=4` line: the AF answers 204, excludes ue3 and runs round 4 before nefd logs the response. The docstring itself said the lines are not ordered against round 4, but the code was. Now bounded below by the Accept only; the amfd "no other leg" check keeps the (Accept, round 4) window where it is what it means. The departure ids (readmission mints new ones) pin each line to this episode.
- `#1492` [e6f4a81f](https://github.com/NextgCoreLab/nextgcore/commit/e6f4a81f) - probe cycle 1 used a regex `"(A|B) "` whose first alternative required a space after `gpsi=`; the real log is `gpsi=extid-ue003@fl.nextg`. Cycles 2-10 passed because the previous cycle's readmit line matched alt B. Drop the trailing space; cycle 1's `member_joined` matches too.
- `#1492` [00d27996](https://github.com/NextgCoreLab/nextgcore/commit/00d27996) - `deploy-eks.sh`'s `verify_images.py` expects every deployed NF Ready in 90 s, but `profile_rollout` (from the 5g profile) does not wait for demo-added NFs like `nef`. The d445a gates arm hit this: nef's `wait-nrf` init container stalled for 2 m 20 s mid-wait, so nef was still in init when verify_images started. The demo profile now wraps `profile_rollout` and waits for every `PROFILE_CORE_NFS` entry not already on the 5g list.
- `#1492` [cec73d25](https://github.com/NextgCoreLab/nextgcore/commit/cec73d25) - the gates arm's `retrieve()` had `local auth=()` and then expanded `"${auth[@]}"`. Under `set -u`, an empty-array expansion is "unbound variable" on some bash builds, so the no-token request sent an empty curl invocation; two gates then reported FAIL for an unrelated reason. Fixed to the `${auth[@]+"${auth[@]}"}` idiom.
- `#1492` [1827d9cd](https://github.com/NextgCoreLab/nextgcore/commit/1827d9cd) - the nefd bogus-mode gate's `grep -m1 -oE 'alt1|alt2'` matched both alternatives on one log line; -o printed each match separated by a newline. The emitted gate text then had two lines, and `check_gates` read the second as a new gate row (no PASS/FAIL prefix, verdict FAIL). Pipe the match through `head -1`.

No base-core bug was surfaced; the only observed non-deterministic cluster behaviour was a one-time `PayloadNotForwarded` cause for ue1 on the first eth0 attempt (12 min registration retry before success); the rerun with reused images cleared it.

## Honesty labels and limitations

- Non-normative: the AF, the client, the frames and the aggregation are application logic, out of 3GPP scope (TS 23.501 §5.46.1). The NEF, UDM and AMF steps follow TS 23.502 §4.15.3.2.3 and §4.15.10.
- The data is synthetic and declared so. Learning is measured by the AF against local-only and pooled baselines it computes from the same pinned task.
- Robustness is claimed only against this fixed attacker (UE 005, ×10 sign-flipped deltas).
- Trust and secure aggregation are out of scope. This is not a privacy proof: the row-free frames and the canary scan show that no training row crosses the network, nothing more.
- UE ID retrieval follows the Rel-18 TS 29.522 OpenAPI (`TS29522_UEId.yaml`, 3gpp-ueid 1.1.1): `POST /3gpp-ueid/v1/retrieve` with a `UeIdReq`, answered by a `UeIdInfo` whose required `externalId` is the AF-specific External Identifier (TS 23.502 §4.15.10). The AF reads no other member. T8 names the UE by that `externalId` (TS 29.122 `MonitoringEventSubscription`), so nefd resolves it back to the UE. TS 29.522 is not vendored in nextgcore, so this is checked against the published OpenAPI only.
- T8 is not claimed to be faster than a TCP keepalive. The AF simply does not treat a closed transport as a departure.
- The AF has no `/metrics` listener (as with D444A's sink). Its demo state is in the startup banner and the `[demo:d445a-fl-t8]` lines.
- One FL client per `nr-ue` process (the first UE instance): the overlay runs one UE per pod.
