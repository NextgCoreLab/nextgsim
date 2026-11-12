# Demo D444A: app adaptation to a network-signalled Session-AMBR (nextgsim#477)

**Non-normative.** The UE application, its feature codec, the wire datagram and
`semantic-sink` are research constructs with no 3GPP Stage-3 behind them. The
normative part is the core's: a PCF-changed Session-AMBR reaches the UE in a PDU
SESSION MODIFICATION COMMAND (TS 23.502 §4.3.3.2 step 3b, TS 24.501 §6.3.2.2,
nextgcore#1385), and the UE's NAS applies it and reports it upward (#454).

| File | What |
|---|---|
| `ue.yaml` | `config/ue.yaml` plus `semantic_comm_enabled`, the `semantic_stream:` block and the `demo:` block |
| `sink.yaml` | `semantic-sink`, bound to 10.45.0.1 inside the UPF's network namespace |
| `compose.override.yml` | over nextgcore's standalone `docker/rust/docker-compose.demo-n2ho.yml`: one gNB, this UE config, and the sink in `network_mode: service:upf` |

Build: `nr-ue` with `--features demo-d444a-ambr-adapt`, `semantic-sink` with
`-p nextgsim-semantic --features demo-d444a-ambr-adapt`.

The throttle is the core's. With nextgsim#465 merged it is that demo's pcfd
controller; until then the EKS scenario lowers the subscriber's session AMBR
in the UDR and re-authorises the session (see
`deploy/eks/overlays/demo-d444a-ambr-adapt/` in nextgcore).

Honesty labels:
- the accuracy gap between keep 40 and the baselines is the pinned generator's
  spectrum (ground truth), not a semantic-communication result;
- "raw is policed at 64 kbps" is arithmetic: 128 features at 50 Hz is
  120 kbit/s of IP;
- the baselines (top-variance raw dimensions, the fair one, and mean-pool, an
  ablation) are scored by the sink on the same received samples at the same
  keep, hence equal bytes; they are not transmitted.

Revert arms: `scripts/demo/d444a-reverts.sh` (code), `follow_session_ambr:
false` and removing `device:` (config).
