# Changelog

All notable changes to **nextgsim** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Scope of validation.** nextgsim is a research/interop 5G UE + gNB simulator validated
> against our own `nextgcore` (matched-sim), **not** against a real gNB/UE/core or a
> conformance suite. The 6G research crates (ISAC / FL / SHE / NKEF / agent) are prototypes,
> disabled by default; there is no frozen 3GPP Rel-20 stage-3 spec.

## [Unreleased] — proposed v0.1.0

First release aligning the simulator with the 3GPP Rel-15/16/17/18 procedures so it
interoperates end-to-end with `nextgcore`. Zero C dependencies; derived from UERANSIM.

### Added
- **gNB AS security** at Initial Context Setup: KgNB → K_RRCint/K_RRCenc, algorithm
  selection, RRC SecurityModeCommand on SRB1 (TS 33.501 A.8 / TS 38.331 5.3.4).
- **gNB DRB** via structured RRCReconfiguration (RadioBearerConfig + SDAP + mapped QoS flows
  + CellGroupConfig) on PDU-session setup (TS 38.331 / 38.300).
- **SRB1** RadioBearerConfig / CellGroupConfig wired into RRCSetup; RRC transaction-id hygiene.
- **AMF-initiated NGAP** procedures (Paging, NG Reset, AMF Config Update, Error Indication) and
  ICS-carried PDU sessions.
- **UE EAP-AKA′** primary authentication.
- **UE UPDP** handling: decode MANAGE UE POLICY COMMAND, store policy sections, answer
  COMPLETE/REJECT (TS 24.501 Annex D).
- Optional native **kernel-SCTP** N2 transport (loopback-validated, no Open5GS).

### Fixed
- Little-endian PSI bitmap on the wire (TS 24.501 §9.11.3.44).
- 38.415 PDU Session Container QFI/RQI wire layout; NH FC value.
- Masked-sum cancellation under unequal sample weights (FL demo).

### Changed
- 6G research crates relabeled as prototypes with honest scope notes + TR 22.837/22.870
  traceability; corrected over-claims (SecAgg crypto strength, DP accounting, ONNX "compliance").
- Documentation synced (README, architecture, nextgcore-integration with correct core IPs).

### Validation
- `cargo test --workspace`: 3273 passed / 0 failed · `clippy`: 0 errors · `fmt`: clean ·
  matched-sim E2E with nextgcore: green.

## [0.0.1] — 2026-03-08
- Initial public tag.
