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

Not yet run on devtest1. This section is filled from the captured artefacts (`/tmp/eks-val/demos/d445a-fl-t8/<mode>/`) after the EKS runs; nothing here is invented ahead of them.

## Honesty labels and limitations

- Non-normative: the AF, the client, the frames and the aggregation are application logic, out of 3GPP scope (TS 23.501 §5.46.1). The NEF, UDM and AMF steps follow TS 23.502 §4.15.3.2.3 and §4.15.10.
- The data is synthetic and declared so. Learning is measured by the AF against local-only and pooled baselines it computes from the same pinned task.
- Robustness is claimed only against this fixed attacker (UE 005, ×10 sign-flipped deltas).
- Trust and secure aggregation are out of scope. This is not a privacy proof: the row-free frames and the canary scan show that no training row crosses the network, nothing more.
- UE ID retrieval follows the Rel-18 TS 29.522 OpenAPI (`TS29522_UEId.yaml`, 3gpp-ueid 1.1.1): `POST /3gpp-ueid/v1/retrieve` with a `UeIdReq`, answered by a `UeIdInfo` whose required `externalId` is the AF-specific External Identifier (TS 23.502 §4.15.10). The AF reads no other member. T8 names the UE by that `externalId` (TS 29.122 `MonitoringEventSubscription`), so nefd resolves it back to the UE. TS 29.522 is not vendored in nextgcore, so this is checked against the published OpenAPI only.
- T8 is not claimed to be faster than a TCP keepalive. The AF simply does not treat a closed transport as a departure.
- The AF has no `/metrics` listener (as with D444A's sink). Its demo state is in the startup banner and the `[demo:d445a-fl-t8]` lines.
- One FL client per `nr-ue` process (the first UE instance): the overlay runs one UE per pod.
