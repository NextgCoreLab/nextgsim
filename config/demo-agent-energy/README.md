# Demo D446A: load-balancing vs energy-saving agents over a booster cell

nextgsim#483, demo key `d446a-agent-conflict`. The convention is in #458.

| file | node | role |
|---|---|---|
| `gnb-a.yaml` | gNB-A, NCI 0x020, gNB ID 2, PCI 32 | coverage cell; agent `lb-1` (load balancing) |
| `gnb-b.yaml` | gNB-B, NCI 0x010, gNB ID 1, PCI 16 | booster; agent `energy-1` (energy saving); holds the **mode** |
| `ue.yaml` | UE | camps on gNB-A, hears both |

Modes (`demo.d446a-agent-conflict.mode` on gNB-B):

- `load-balancing`: `lb-1` wins `Cell:16`. One N2 handover A to B with cause
  "Reduce load in serving cell". `energy-1` is Deferred, then held by the
  `SwitchOffWithConnectedUes` guardrail. B holds PCI 32 against an A3 handback.
- `energy-saving`: `energy-1` wins. B answers HANDOVER FAILURE "Cell not
  available", the AMF answers A with HANDOVER PREPARATION FAILURE, and B goes
  off the air: RRC barred, RLS heartbeats unanswered.

Switch at runtime: `nr-cli <gnb-b> demo d446a-agent-conflict set mode=<mode>`.
The change applies to the next contest, and the audit records it as
`PolicyChanged`. Read the audit with `nr-cli <gnb-b> agent-audit --json`.

Run it with `scripts/demo/d446a-check.sh <mode|off>`. The gNB must be built
with `--features demo-d446a-agent-conflict`.

## Honesty labels

- **Non-normative policy.** TS 38.300 §15.5.1.3 leaves admission control for
  load-balancing handovers to the target cell. Refusing an inbound handover
  while a switch-off is still *Pending* is this demo's policy. §15.4.2.1 lets a
  booster prevent incoming handovers only during the switch-off time period.
- **Xn stand-in.** No Xn is modelled. The NGAP cause carries `lb-1`'s intent.
  The booster learns the source gNB from the UE History Information and maps
  it to `lb-1` through its `peers:` list.
- **Simple inputs.** Both agents are rule-based. Load is a served-UE count, and
  RSRP comes from the simulator's distance model.
- **Baseline.** A measurement-triggered (A3) handover still carries "Handover
  desirable for radio reasons".
