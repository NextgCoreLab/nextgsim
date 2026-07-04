# gNB Architecture

This chapter is a component reference for `nextgsim-gnb`, the 5G/6G gNodeB
simulator binary. It maps the binary's internal modules, the protocol-stack
crates it composes, and the actor/task model that wires them together. Every
structural and behavioural claim below carries a concrete code reference
(crate + file, plus a function or type) that was read directly from the source.
For the peer node, see [UE Architecture](ue.md); for the runtime knobs named
here, see the [gNB section of the Configuration Reference](../configuration.md#gnb-configuration).

> **Honesty note:** the behaviour described here is validated by the crate unit
> tests and by *matched-simulator* end-to-end runs — the project drives its own
> gNB and UE against the sibling `nextgcore` 5G core, and the UE→UPF GTP-U data
> plane has been verified end-to-end. This is **not** third-party conformance
> certification: it proves internal consistency of the nextgsim/nextgcore pair.
> The Rel-17/18 features use a mix of real ASN.1 and sim-internal encodings
> (called out below), and the 6G / AI-native modules (`agent`, `fl`, `isac`,
> `nkef`, `nwdaf`, `she`) are non-normative research prototypes of 3GPP Stage-1
> studies — no frozen Rel-20 Stage-3 wire spec exists to conform to.

## Binary and startup

The runnable binary is declared in `nextgsim-gnb/Cargo.toml` as
`[[bin]] name = "nr-gnb"` (path `src/main.rs`), alongside a library target
`nextgsim_gnb` (`src/lib.rs`). The crate has one Cargo feature, `kernel-sctp`,
which forwards to `nextgsim-sctp/kernel-sctp` for a real kernel-SCTP transport
on Linux; it is off by default.

`src/main.rs` is a `#[tokio::main]` async entry point. Its flow:

1. `init_logging()` installs a `tracing_subscriber` registry plus a
   `tracing_log::LogTracer` bridge, so records emitted through the `log` crate
   (for example the ISL-handover path in `rrc/ntn_gnb.rs`) land in the same
   pipeline (`src/main.rs`, `init_logging`).
2. `Args` (clap) parses `-c/--config <FILE>` and `-l/--disable-cmd`
   (`src/main.rs`, `struct Args`).
3. `run_gnb()` builds `GnbApp::new()`, then calls `connect_to_amfs()`, then
   `run()` (waits for shutdown), then `shutdown()` (`src/main.rs`, `run_gnb`).

`GnbApp::new()` (`src/main.rs`) does the real wiring:

1. `load_and_validate_gnb_config(path)` parses and validates the YAML into a
   `GnbConfig` (re-exported from `app::config_loader`, `src/app/mod.rs`).
2. `TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY)` creates every task's
   mpsc channel and hands back the six core receivers (`src/tasks.rs`,
   `TaskManager::new`).
3. `spawn_tasks()` `tokio::spawn`s each task's `run` loop.

`connect_to_amfs()` iterates `config.amf_configs` and, for each, sends the SCTP
task a `SctpMessage::ConnectionRequest` carrying `ppid: NGAP_PPID` (=`60`) and a
per-AMF `client_id` (`src/main.rs`, `connect_to_amfs`). Nothing is sent on the
wire until the SCTP association comes up — the NG Setup that follows is driven
by the NGAP task (see [Internal wiring](#internal-wiring--task-model)).

The App task is spawned first; unless `--disable-cmd` is set it calls
`init_cli_server("gnb")` to open the UDP CLI IPC port that `nr-cli` connects to
(`src/main.rs`, `spawn_tasks`; see the [nr-cli Reference](../cli-reference.md)).

## Module map

The gNB binary's own modules (`src/lib.rs` declares them all):

| Module | Path | Responsibility |
|---|---|---|
| `app` | `src/app/` | Config load/validate (`config_loader`), UDP CLI server (`cli_server`), command parsing/handling (`cmd_handler`), status reporting (`status`), App task loop (`task`). Central coordinator. |
| `tasks` | `src/tasks.rs` | Actor framework: `Task` trait, `GnbTaskBase`, `TaskHandle`, `TaskManager`, all inter-task message enums, and the constants (`DEFAULT_CHANNEL_CAPACITY = 256`, `NGAP_PPID = 60`). |
| `ngap` | `src/ngap/` | N2 control plane: `NgapTask` (`task.rs`), AMF context (`amf_context.rs`), UE context (`ue_context.rs`), MBS context (`mbs_context.rs`). |
| `rrc` | `src/rrc/` | Air-interface control: `RrcTask` (`task.rs`), connection/state management (`connection.rs`, `ue_context.rs`), transactions, handover, redcap, energy-saving, NTN (`ntn_gnb.rs`). |
| `rls` | `src/rls/` | `RlsTask`: UDP radio-link simulation — UE discovery, RRC relay, user-plane relay. |
| `gtp` | `src/gtp/` | `GtpTask`: N3 GTP-U user-plane tunnel management and forwarding. |
| `sctp` | `src/sctp/` | `SctpTask` and `AmfConnection`: N2 transport (association management, backend selection). |
| `energy` | `src/energy/` | `EnergyTask` (Rel-18): cell sleep/dormant modes and energy-efficiency metrics. Spawned only if `energy_saving_enabled`. |
| `daps` | `src/daps.rs` | Rel-17 DAPS handover types (`DapsManager`, `DapsState`), per code comments TS 38.300 §10.1.2.4. Library module — see [gaps](#simplifications--known-gaps). |
| `mbs_ngap` | `src/mbs_ngap.rs` | Rel-17 MBS NGAP session types (`NgapMbsManager`, `GnbMbsState`), per code comments TS 38.413 §8.21. Library module — see [gaps](#simplifications--known-gaps). |
| `agent` | `src/agent/` | 6G AI Agent Framework task. One-line honest pointer only; deferred to the [AI/6G stack chapter](../concepts/ai-6g-stack.md). |
| `fl` | `src/fl/` | 6G Federated-Learning aggregator task. Prototype — see [AI/6G stack](../concepts/ai-6g-stack.md). |
| `isac` | `src/isac/` | 6G Integrated Sensing & Communication task. Prototype — see [AI/6G stack](../concepts/ai-6g-stack.md). |
| `nkef` | `src/nkef/` | 6G Network Knowledge Exposure Function task. Prototype — see [AI/6G stack](../concepts/ai-6g-stack.md). |
| `nwdaf` | `src/nwdaf/` | 6G Network Data Analytics task. Prototype — see [AI/6G stack](../concepts/ai-6g-stack.md). |
| `she` | `src/she/` | 6G Service Hosting Environment task. Prototype — see [AI/6G stack](../concepts/ai-6g-stack.md). |

Each AI/6G module is a thin wrapper: `<name>/mod.rs` re-exports a `<name>/task.rs`
that adapts the corresponding `nextgsim-<name>` crate into the gNB task
framework (e.g. `src/nkef/mod.rs` header: "wires the nextgsim-nkef crate into
the gNB task framework"). Their message types live in `src/tasks.rs`
(`SheMessage`, `NwdafMessage`, `NkefMessage`, `IsacMessage`, `AgentMessage`,
`FlAggregatorMessage`). Detail belongs in the [AI/6G stack chapter](../concepts/ai-6g-stack.md);
this chapter treats them as optional, config-gated side tasks.

## Protocol stack

The control- and user-plane logic lives in standalone crates listed under
`[dependencies]` in `nextgsim-gnb/Cargo.toml`. The task modules above are
adapters that own sockets, contexts, and message loops, and call into these
crates for codec and procedure work.

### NGAP (N2 control plane) — `nextgsim-ngap`

- **Codec:** `nextgsim-ngap/src/codec.rs` provides `encode_ngap_pdu` /
  `decode_ngap_pdu` over **Aligned PER (APER)** using `asn1_codecs::aper`. The
  Rust types (`NGAP_PDU` and friends) are generated from the 3GPP NGAP ASN.1
  schema at build time via `include!(concat!(env!("OUT_DIR"), "/ngap.rs"))`.
  This is real ASN.1/APER, not a bespoke encoding. `encode_aper`/`decode_aper`
  handle the opaque per-session transfer containers.
- **Procedures:** `nextgsim-ngap/src/procedures/` (see `procedures/mod.rs`,
  header cites TS 38.413) implements NG Setup, Initial UE Message, NAS transport
  (up/downlink), Initial Context Setup, PDU Session Resource setup/modify/release,
  the transfer containers, UE Context Release, Paging, Path Switch, Handover, and
  NG Reset — plus 6G extension modules (`ai_native`, `isac_reporting`,
  `ntn_support`).
- **What the gNB does with it:** `NgapTask` (`src/ngap/task.rs`) imports these
  procedure builders/parsers directly. `send_ng_setup_request` derives the gNB
  ID as `nci >> (36 - gnb_id_length)` and advertises `ran_node_name =
  "nextgsim-gnb"` and a supported-TA list built from `config.nssai`.
  `handle_initial_nas_delivery` builds `InitialUeMessageParams` and calls
  `encode_initial_ue_message`. AS security at Initial Context Setup is real:
  `select_as_algorithms` picks NEA/NIA from the UE Security Capabilities bitmaps
  (preferring AES/NEA2/NIA2), and `activate_as_security` derives the four AS keys
  from KgNB via `nextgsim_crypto::kdf` (per code comments TS 33.501 Annex A.8).

### RRC (air-interface control) — `nextgsim-rrc`

- **Codec:** `nextgsim-rrc/src/codec.rs` provides `encode_rrc` / `decode_rrc`
  over **Unaligned PER (UPER)** using `asn1_codecs::uper`, with types generated
  from the 3GPP RRC ASN.1 schema (`OUT_DIR/rrc.rs`). Core messages (RRC Setup
  Request/Setup/Setup Complete, DL Information Transfer, Security Mode Command,
  RRC Reconfiguration, UE Capability) use this real UPER path.
- **What the gNB does with it:** `RrcTask` (`src/rrc/task.rs`) dispatches uplink
  PDUs by logical channel in `handle_uplink_rrc` — UL-CCCH via
  `decode_rrc_setup_request`, UL-DCCH via `dispatch_ul_dcch`. Downlink NAS is
  wrapped with `encode_dl_information_transfer` and relayed to RLS. The
  `SecurityModeCommand` and `RrcReconfiguration` PDUs arrive already ASN.1-encoded
  from the NGAP task and are relayed on DL-DCCH.
- **Real vs simplified:** some paths deliberately fall back to sim-internal
  encodings for interop with the matched UE. `src/rrc/task.rs` defines
  `RRC_MSG_TYPE_UE_CAPABILITY = 0x06` as an envelope byte in front of the real
  ASN.1; `src/rrc/connection.rs` notes the gNB-side RRCResume (DL-CCCH) "uses a
  simplified encoding" because the ASN.1 gNB-side builder requires
  `masterCellGroup` and radio-bearer configuration from the suspended UE context;
  and `src/rrc/handover.rs` builds handover RRC Reconfiguration in a "Simplified
  format (not real ASN.1)". These are called out again under
  [gaps](#simplifications--known-gaps).

### RLS (radio-link simulation) — `nextgsim-rls`

RLS is the stand-in for the PHY/over-the-air link; there is no 3GPP wire spec
for it. `nextgsim-rls/src/lib.rs` describes a UDP protocol (UERANSIM-inspired)
with four message types: `Heartbeat`, `HeartbeatAck`, `PduTransmission`,
`PduTransmissionAck`, plus a custom binary `codec` and a `cell_search`
(`GnbCellTracker`) tracker.

- **What the gNB does with it:** `RlsTask` (`src/rls/task.rs`) binds a UDP socket
  on `link_ip:4997` (`DEFAULT_RLS_PORT`) and uses its NCI as the gNB STI.
  Heartbeats drive UE discovery (`GnbCellTracker::process_heartbeat` →
  `UeDetected` → `RrcMessage::SignalDetected`). A `PduTransmission` of type
  `Rrc` becomes `RrcMessage::UplinkRrc`; type `Data` is fed through a per-UE
  **RLC** entity for reassembly before it is forwarded to GTP (below).

### GTP-U (N3 user plane) — `nextgsim-gtp`

- **Codec:** `nextgsim-gtp/src/lib.rs` cites TS 29.281 and exposes `GtpHeader`
  (`codec.rs`), a `TunnelManager` (`tunnel.rs`) and a 5QI/QoS table (`qos.rs`),
  plus 6G extension-header markers (TSN, in-network compute).
- **What the gNB does with it:** `GtpTask` (`src/gtp/task.rs`) binds a UDP socket
  on `gtp_ip:2152` (`GTP_U_PORT`). If `config.upf_addr` is `None` it runs in
  **loopback mode** (echoes user data back to the UE for tests without a UPF);
  otherwise uplink SDUs are `encapsulate_uplink`'d into GTP-U G-PDUs to the UPF,
  and downlink G-PDUs are `decapsulate_downlink`'d in `handle_downlink_gpdu` and
  handed to RLS as `RlsMessage::DownlinkData`. It also answers GTP-U Echo Request.
- **Real vs simplified:** the SDAP/DRB QoS layer is not fully wired — per code
  comments in `handle_downlink_gpdu` (amfg-04/amfg-09), the session is selected
  by **PSI** and the DL **QFI/RQI** are surfaced only for diagnostics /
  reflective QoS rather than driving DRB selection.

### SCTP (N2 transport) — `nextgsim-sctp`

- **Backend:** `nextgsim-sctp/src/lib.rs` provides SCTP over a userspace
  `sctp-proto` engine running on tokio UDP, and states it is wire-compatible with
  nextgcore's SCTP (both use SCTP-over-UDP). `SctpTask` (`src/sctp/task.rs`)
  selects the backend from `config.sctp_backend`: `SctpBackendKind::Userspace`
  (the default, per `nextgsim-common/src/config.rs`) or `SctpBackendKind::Kernel`
  (real IP proto 132, only on Linux builds with the `kernel-sctp` feature).
- **Association:** `src/sctp/amf_connection.rs` wraps the association in an
  `AssociationKind` enum — `Single`, `Multihome` (used when an AMF config carries
  `secondary_addresses`, via `MultihomeSctpAssociation`), or `Kernel`.
- **What the gNB does with it:** the SCTP task maps `ConnectionRequest` → connect,
  routes inbound peer data up to the NGAP task as
  `NgapMessage::ReceiveNgapPdu`, and surfaces association up/down. A QUIC
  transport exists (`nextgsim-sctp/src/quic.rs`) but is **not** selected: if
  `quic_enabled` is set the task logs a warning and falls back to SCTP
  (`src/sctp/task.rs`, `handle_connection_request`).

### Supporting crates

- **`nextgsim-rlc`** (`src/lib.rs`, cites TS 38.322): RLC TM/UM/AM entities. The
  gNB RLS task creates one **UM, SN12** `RlcEntity` per UE for user-plane
  reassembly (`src/rls/task.rs`, `rlc_entity_for`).
- **`nextgsim-common`** (`src/config.rs`, `types.rs`): `GnbConfig`, the
  `TaskMessage<T>` envelope, `OctetString`, `Plmn`, and the SCTP backend enum.

## Internal wiring / task model

The gNB is an **actor system**: each component is an independent `tokio` task
that owns its state and communicates only by typed messages over `tokio::mpsc`
channels. The contract is the `Task` trait (`src/tasks.rs`):

```rust
#[async_trait::async_trait]
pub trait Task: Send + 'static {
    type Message: Send;
    async fn run(&mut self, rx: mpsc::Receiver<TaskMessage<Self::Message>>);
}
```

`TaskMessage<T>` (defined in `nextgsim-common/src/types.rs`) is a two-variant
envelope — `Message(T)` or `Shutdown` — so every task's `run` loop drains one
channel and exits cleanly on `Shutdown`.

**Handles and the shared base.** `GnbTaskBase` (`src/tasks.rs`) is a cloneable
struct holding an `Arc<GnbConfig>` plus one `TaskHandle<_>` per task
(`app_tx`, `ngap_tx`, `rrc_tx`, `gtp_tx`, `rls_tx`, `sctp_tx`), and optional
`sixg` / `rel18` handle bundles. A `TaskHandle<T>` wraps an `mpsc::Sender`; its
`send()` wraps the payload in `TaskMessage::Message`, and `shutdown()` sends
`TaskMessage::Shutdown`. Every task gets a clone of the base, so any task can
message any other by name — this is the "wiring".

**Channel topology.** `GnbTaskBase::new` creates all six core channels at
`DEFAULT_CHANNEL_CAPACITY = 256` (`src/tasks.rs`). The 6G and Rel-18 channels are
created lazily by `init_6g_tasks()` / `init_rel18_tasks()` only when those
features are enabled (`src/main.rs`, `spawn_tasks`).

The signalling data flow across tasks:

```text
        AMF (N2)                                   UE (RLS/UDP)
          │  SCTP                                      │  UDP
          ▼                                            ▼
     ┌──────────┐   NgapMessage    ┌──────────┐  RrcMessage  ┌──────────┐
     │  SCTP    │ ───────────────► │   NGAP   │ ───────────► │   RRC    │
     │  task    │ ◄─────────────── │   task   │ ◄─────────── │   task   │
     └──────────┘   SctpMessage    └────┬─────┘  NgapMessage └────┬─────┘
                                        │ GtpMessage              │ RlsMessage
                                        ▼                         ▼
                                   ┌──────────┐  RlsMessage  ┌──────────┐
                                   │   GTP    │ ───────────► │   RLS    │
                                   │   task   │ ◄─────────── │   task   │
                                   └────┬─────┘  GtpMessage  └────┬─────┘
                                        │ GTP-U/UDP               │ UDP
                                        ▼                         ▼
                                      UPF (N3)                   UE
```

**Control-plane path (registration).** Numbered, each step a real handler:

1. UE heartbeat over UDP → RLS discovers it and sends
   `RrcMessage::SignalDetected` (`src/rls/task.rs`, `handle_heartbeat`).
2. UE RRC Setup Request (UL-CCCH) → `RrcTask::handle_uplink_rrc` →
   `decode_rrc_setup_request` (`src/rrc/task.rs`).
3. RRC Setup Complete carries the initial NAS PDU → RRC sends
   `NgapMessage::InitialNasDelivery` to NGAP.
4. `NgapTask::handle_initial_nas_delivery` allocates a RAN-UE-NGAP-ID, builds
   `InitialUeMessageParams`, `encode_initial_ue_message`, and hands the APER
   bytes to the SCTP task as `SctpMessage::SendMessage` (`src/ngap/task.rs`).
5. Downlink NAS from the AMF arrives as `SctpMessage::ReceiveMessage` →
   `NgapMessage::ReceiveNgapPdu` → `handle_ngap_pdu` (`decode_ngap_pdu`) →
   `RrcMessage::NasDelivery` → RRC wraps it in DL Information Transfer → RLS →
   UE.

The full registration walkthrough is in
[Registration flow](../concepts/registration-flow.md).

**User-plane path (PDU session).** Once a PDU session is set up:

1. Uplink: UE data PDU over RLS → `RlsTask::handle_uplink_data` feeds the per-UE
   RLC (UM/SN12) entity, and each reassembled SDU becomes
   `GtpMessage::DataPduDelivery` (`src/rls/task.rs`).
2. `GtpTask` `encapsulate_uplink`s the SDU into a GTP-U G-PDU and sends it to the
   UPF (`src/gtp/task.rs`).
3. Downlink: G-PDU from the UPF → `handle_downlink_gpdu` decapsulates →
   `RlsMessage::DownlinkData` → RLS transmits it to the UE.

The full user-plane walkthrough is in
[PDU session & user plane](../concepts/pdu-session-userplane.md).

**Startup and shutdown.** `TaskManager` (`src/tasks.rs`) owns the channels, a
`watch::channel<bool>` shutdown broadcast, and a per-task `TaskState` map
(`Created → Running → Stopping → Stopped/Failed`). `TaskManager::shutdown()`
signals the watch channel and calls `GnbTaskBase::shutdown_all()`, which sends a
`Shutdown` message to every task handle, then waits with a
`DEFAULT_SHUTDOWN_TIMEOUT_MS = 5000` deadline. One honest structural nuance:
in the live binary, `src/main.rs` spawns each task with a bare `tokio::spawn`
and never calls `register_task_handle` or `mark_task_started`, so at runtime the
manager's join-handle map is empty and its state machine is exercised mainly by
the unit tests in `src/tasks.rs`; production shutdown is driven by the
`Shutdown` broadcast plus the watch signal that `main.rs` selects on in
`GnbApp::run`.

## Simplifications & known gaps

All of the following are grounded in the current source, not in spec text:

- **NTN is parse-forward-store but effectively inert.** When an NG Setup Response
  arrives and `config.ntn_config` is set, the NGAP task logs it and forwards
  `RrcMessage::NtnTimingAdvanceConfig` to RRC (`src/ngap/task.rs`,
  `handle_ng_setup_response`). `RrcTask` stores it into `self.ntn_config`
  (`src/rrc/task.rs`) — but that stored value is only ever *written*, never read
  back to shape an outgoing RRC Setup/Reconfiguration. The
  [Configuration Reference](../configuration.md#ntn-research-prototype) documents
  `ntn_config` as logging-only/inert for exactly this reason. The NTN modules
  under `src/rrc/ntn_gnb.rs` are research prototypes and use "Simplified"
  distance/beam math per their own code comments.
- **DAPS and standalone MBS-NGAP are library modules, not wired into a task.**
  `src/daps.rs` (`DapsManager`, `DapsState`) and `src/mbs_ngap.rs`
  (`NgapMbsManager`, `GnbMbsState`) are re-exported from `src/lib.rs` but are not
  instantiated in any task `run` loop. MBS activation/deactivation *is* handled
  live, but through a different type — `MbsSessionManager` in
  `src/ngap/mbs_context.rs`, driven by the `MbsSession*` variants of
  `NgapMessage` in the NGAP task loop.
- **QUIC transport is present but unreachable.** `nextgsim-sctp/src/quic.rs`
  exists, yet `SctpTask::handle_connection_request` (`src/sctp/task.rs`) warns and
  falls back to SCTP when `quic_enabled` is set — QUIC is never actually selected.
- **Some RRC encodings are sim-internal.** The UE-capability envelope byte
  (`RRC_MSG_TYPE_UE_CAPABILITY = 0x06`, `src/rrc/task.rs`), the gNB-side
  RRCResume DL-CCCH fallback (`src/rrc/connection.rs`), and the handover RRC
  Reconfiguration "not real ASN.1" format (`src/rrc/handover.rs`) are simplified
  for interop with the matched UE rather than being fully spec-conformant on the
  wire.
- **User-plane QoS below the flow level is not modelled.** As noted above, GTP-U
  downlink selects the session by PSI and treats QFI/RQI as diagnostic; there is
  no SDAP-to-DRB mapping yet (`src/gtp/task.rs`, `handle_downlink_gpdu`).
- **6G / AI-native tasks are prototypes.** `agent`, `fl`, `isac`, `nkef`,
  `nwdaf`, `she` are each config-gated (`*_enabled`) and only spawned when
  enabled (`src/main.rs`, `spawn_tasks`). They implement Stage-1 study ideas, not
  standardised procedures — see the [AI/6G stack chapter](../concepts/ai-6g-stack.md).

## Where to look next

- [Configuration Reference — gNB section](../configuration.md#gnb-configuration):
  the YAML fields (`nci`, `gnb_id_length`, `plmn`, `nssai`, `amf_configs`,
  interface IPs, `ntn_config`) that feed the `GnbConfig` used throughout this
  chapter.
- [nr-cli Reference](../cli-reference.md): the `info`/`status`/`amf-list`/
  `ue-list`/`ue-info`/`ue-release` commands the App task serves.
- [Registration flow](../concepts/registration-flow.md): the end-to-end control
  path that threads RLS → RRC → NGAP → SCTP → AMF.
- [PDU session & user plane](../concepts/pdu-session-userplane.md): the RLC →
  GTP-U → UPF data path.
- [AI/6G stack](../concepts/ai-6g-stack.md): detail on the six prototype tasks
  summarised in the module map.
- [UE Architecture](ue.md): the matched peer that drives this gNB.
- [6G gap analysis](../gaps/6g-gap-analysis.md) and
  [protocol gaps](../gaps/gap-protocols.md): the broader honest inventory of what
  is and isn't implemented.
