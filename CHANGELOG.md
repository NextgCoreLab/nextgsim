# Changelog

All notable changes to **nextgsim** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Scope of validation.** nextgsim is a research/interop 5G UE + gNB simulator validated
> against our own `nextgcore` (matched-sim), **not** against a real gNB/UE/core or a
> conformance suite. The 6G research crates (ISAC / FL / SHE / NKEF / agent) are prototypes,
> disabled by default; there is no frozen 3GPP Rel-20 stage-3 spec.

## [Unreleased]

_Nothing yet._

## [0.2.0] — Unreleased (draft for operator review)

Draft built from the PRs merged into `main` since the 0.1.0 tag (2026-07-02), cross-checked
against `git log 0.1.0..main`. Every bullet cites its PR; the operator reviews this section
before the tag. The validation line is filled from the release run, not carried over.

### Highlights
- **A real LTE RAN.** New crates `nextgsim-enb`, `nextgsim-lte-ue`, `nextgsim-lte-rrc`,
  `nextgsim-eps-nas` and `nextgsim-s1ap`: an eNB that completes S1 Setup, E-UTRA RRC, EPS NAS
  attach with AS security, and an LTE user plane over PDCP/RLC UM and S1-U GTP-U into a UE TUN
  (#221, #225, #236, #239), with container images and a ping self-test (#240).
- **ASN.1 from NextgCoreLab/asn1-codegen.** The ASN.1 compiler and codecs are consumed from
  asn1-codegen v0.1.0 and the vendored `vendor/asn1-*` copies are gone (#263).
- **Paging and Service Request.** PCCH paging end to end (#100), paging occasion from a shared
  frame clock (#130), and a Service Request from CM-IDLE per TS 24.501 §4.4.6 that answers a page
  (#259), holding the triggering uplink packet until the user plane is back (#274); Service
  Request ngKSI/service-type half-octet order and Notification access type (#261).
- **RACS on the RAN side**: capability cache, UE Radio Capability ID Mapping and Info Indication,
  the 5GMM RACS bit (#249, #251, #255).
- **Restart and recovery** for the EKS runs: hostname peers, AMF reconnect with re-NG-Setup
  (#289), a gNB restarted at a new IP is adopted (#294), `Recreate` strategy for gNB/UE
  Deployments (#291).

### LTE (eNB, LTE UE, EPS NAS, S1AP)
- Generated S1AP codec and S1 Setup (#221); E-UTRA RRC connection and NAS relay over RLS (#225);
  EPS NAS attach, AS security and Initial Context Setup (#236); SRB2, DRBs on PDCP/RLC UM, S1-U
  GTP-U and the UE TUN (#239); `Dockerfile.enb-local` / `Dockerfile.lte-ue-local`, env overrides
  and a ping self-test (#240). Tracked as nextgsim #186.

### ASN.1 codecs
- Consumed from NextgCoreLab/asn1-codegen v0.1.0, `vendor/asn1-*` dropped (#263).
- Fixes made before the move upstream: CHOICE extension-arm encoding (#181), ENUMERATED extension
  additions (#216), opt-in skipping of unknown IEs with criticality (#217), SEQUENCE OF SIZE
  (#215), empty extension-set entries (#219), fragment lengths of 16K and over (#223), variable-size
  BIT/OCTET STRING alignment (#227), extensible INTEGER past the root (#235), mixed range+value
  constraint bounds (#242), character-string alignment (#247), SEQUENCE extension additions (#256).
- Generated NRPPa codec answering the three positioning procedures (#195); Rel-19 RRC ASN.1
  vendored (#184); residual hand-rolled RRC PDUs turned into UPER (#146); property-based codec
  round trips with proptest (#109).

### gNB
- NGAP: UE Context Modification with AS re-keying (#67), Error Indication handling (#69),
  Overload Start/Stop (#70), AMF Status Indication (#71), AMF Configuration Update (#72), RAN
  Configuration Update (#73, #145), slice-aware AMF selection (#74), GUAMI/AMF-set and multi-TNLA
  (#75), guard timers (#92), PDU Session Resource Notify (#144), MBS session procedures (#187,
  #194), TS 38.413 §10.3.4 criticality handling (#245, #269), message-level NAS-PDU of the Setup
  Request delivered after the DRB (#298).
- GTP-U Error Indication, Recovery counter and N3 path supervision (#90); purge a restarted GTP-U
  peer's tunnels (#178).
- RedCap restriction set and the 20 MHz ceiling (#63, #179, #193); SIB19 and NTN timing advance
  (#189); MIB/SIB1 on BCCH (#108); measConfig from the generated codec (#182); QoS flows to DRBs
  with SDAP (#183); AS keys handed to RRC before the SecurityModeCommand (#287).
- CLI: the gNB converges on the registering CliServer so `nr-cli` reaches it (#198); `ue-list`
  and `ue-suspend` work on real UEs (#200).

### UE
- 5GMM: network-initiated de-registration (#60, #273, #282), SUCI reuse under T3519 (#61), PDU
  session status and Uplink data status (#93), Annex C.1 non-volatile storage (#94), UE STATE
  INDICATION and Payload container (#95), Configuration Update Command (#102), radio capability
  ID in the REGISTRATION REQUEST (#121), UUAA-MM container (#125), KgNB from the SMComplete COUNT
  (#204), an integrity-protected Registration Request from 5GMM-IDLE (#260), T3510 expiry releases
  the NAS connection (#279), loss of identity at the network releases sessions before
  re-registering (#296).
- URSP rule evaluation (#96) and uplink traffic classification for it (#131).
- RRC: library RrcTask driven by the UE binary with T300 (#76, #77), PLMN selection from the live
  report (#78, #134), idle-mode reselection parameters and limited service (#143), measurements
  A6/B1/B2 and leaving conditions (#114, #115, #116, #118), re-establishment shortMAC-I check
  (#119), RRC_INACTIVE and authenticated resume (#149, #202), T300/T301/T311/T319 from the
  broadcast (#177), T310 (#175), T304 (#172), guard-timer ticking (#159), DCCH/CCCH dispatch on
  the decoded message (#152), cell reselection after radio link failure (#276).
- CLI: `status` reports the real CM state (#281); `deregister` sends a UE-originated
  DEREGISTRATION REQUEST (#271).

### Mobility
- Inter-gNB handover with re-keying (#150), conditional handover armed from the gNB and driven
  end to end (#161, #163, #166), handover arrival reported to the AMF (#157, #173).

### Security and user plane
- AS security for real: one shared SRB PDCP layer, SMC verification, NEA3 (#147); DRB user data
  protected with K_UPenc/K_UPint per the SMF's SecurityIndication (#148); DRB PDCP entity (#132,
  new crate `nextgsim-pdcp`).
- RLC: per-bearer entities and the UM reassembly window (#104), AM ARQ (#106), UMD SN only when
  segmented (#133).
- PQC primitives held to NIST ACVP vectors and gated off (#192); the `pqc` YAML key honoured and
  unimplemented algorithms dropped (#84).

### Positioning, sidelink and relay
- LPP requests from the LMF answered (#135); sidelink ranging pipeline with SL-PRS stimulus, report
  to the LMF over LPP, CI run (#153, #154, #155) and the LPP EPDU form (#262).
- Real PC5 procedures (#191); L2 UE-to-Network relay with SRAP (#196).

### SCTP
- The shared SCTP core comes from nextgcore, the client half stays here (#210); the patched
  `sctp-proto` comes from nextgcore and the vendored copy is dropped (#234); structs taken from
  libc (#211).
- Userspace backend: PPID carried verbatim (#224), every message on a stream delivered (#229).
- Kernel backend: PPID warning per association (#233), libsctp's int ABI honoured and partial
  deliveries reassembled (#248).

### CI and release
- Nightly interop E2E against nextgcore (#208); a release is cut by promoting an E2E-validated
  (nextgcore, nextgsim) pair (#207); the Test job no longer OOMs on the generated RRC codecs (#280).

### 6G research prototypes (feature-gated, off by default)
- 6G/AI crates are optional gNB dependencies (#68); in-process pub/sub bus (#122), ONNX model
  registry (#123), measured RLS data into the gNB NWDAF task (#127), ISAC EKF (#128), semantic
  codec traits (#129), pluggable `ChannelModel` in `nextgsim-rls` (#110).
- Honesty fixes: NWDAF (#62, #86, #87), NKEF vector search (#85), ranging and sidelink no longer
  advertised as compliant (#65, #66, #140, #142).

### Removed
- Vendored `vendor/asn1-*` and the vendored `sctp-proto` (#263, #234); the unreachable
  `PerformMmCycle` attach path (#89); the dead RedCap PRB clamp (#63).

### Known limitations / deferred to next release
Issues labelled `post-0.2.0` (open, as of this draft):
- #293 gnb: GTP `auto_create_upf_session` forwards uplink for a UE with no PDU Session Resource
  Setup, with an invented TEID.
- #285 ue: no SRB1 PDCP protection after AS security activation; `as_security_enabled: true` drops
  every UL-DCCH PDU after SecurityModeComplete.
- #283 LTE UE/eNB: no dedicated EPS bearer support.
- #277 ue: a failed mobility/periodic registration update drops to 5GMM-DEREGISTERED instead of
  ATTEMPTING-REGISTRATION-UPDATE.
- #264 ue: cell reselection after radio link failure is incomplete (#276 is a partial fix).

### Validation
- _To be filled from the release run (`cargo test --workspace`, clippy, fmt, nightly E2E pair)._

## [0.1.0] — 2026-07-02

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
