# UE Architecture

This chapter is a component reference for the **UE simulator** — the `nextgsim-ue`
crate that builds the `nr-ue` binary. It maps the binary's internal modules,
names the crates it composes, and traces how the NAS state machine drives
registration and PDU-session setup and how user-plane packets reach the TUN
device. Every structural and behavioural claim below names the file and the
function, struct, or constant that backs it, so you can open the source and read
along. For the message-by-message flows this component *participates in*, see the
[registration flow](../concepts/registration-flow.md) and
[PDU session / user plane](../concepts/pdu-session-userplane.md) concept chapters;
this page is about how the UE itself is wired.

> **Honesty note:** The `nextgsim-ue` behaviour described here is validated by
> per-crate unit tests and by the project's **matched-simulator docker E2E** — the
> project drives *its own* gNB/UE against the NextGCore 5G core, and the UE→UPF
> GTP-U data plane is exercised end to end (84/84 green, 0% loss as of 2026-07-02).
> This proves internal consistency of the pair; it is **not** third-party
> conformance certification, and the peer is a cooperative simulator, not an
> arbitrary commercial network. The live authentication path supports **5G-AKA and
> EAP-AKA′** (both Milenage-based); the AS-security wire path is **default-off** (see
> *Simplifications & known gaps*). The Rel-18 5G-Advanced tasks and the 6G/AI
> client tasks are **non-normative research prototypes** — no frozen Rel-20
> Stage-3 wire spec exists. 3GPP TS numbers below are quoted only where they
> appear in this repo's source comments, phrased "per code comments".

## Binary and startup

The crate declares one binary, `nr-ue`, in `nextgsim-ue/Cargo.toml`
(`[[bin]] name = "nr-ue", path = "src/main.rs"`). Startup lives in
`nextgsim-ue/src/main.rs`:

1. **Parse CLI.** `Args` (clap) accepts `-c/--config` (required), `-i/--imsi`
   (override), `-n/--num-of-UE` (1–512), `-t/--tempo` (ms between UE starts),
   `-l/--disable-cmd`, and `-r/--no-routing-config`. `Args` is validated into
   `UeOptions` via `TryFrom<Args>`; `normalize_imsi` strips an `imsi-` prefix and
   enforces exactly 15 digits. These flags are documented for operators in the
   [CLI section of Configuration](../configuration.md#ue-cli-options).
2. **Load + validate config.** `load_and_validate_ue_config` →
   `load_ue_config` (serde_yaml into `nextgsim_common::config::UeConfig`) then
   `validate_ue_config`, which checks HPLMN MCC/MNC ranges, a non-empty
   `gnb_search_list`, a non-zero subscriber key, and performs a **trial**
   `nextgsim_ue::nas::mm::build_suci(config)` so a bad SUCI configuration fails at
   startup rather than inside the NAS task.
3. **Build the app.** `UeApp::new` calls `TaskManager::new` (from `tasks.rs`) to
   mint the four core channels, then `task_manager.init_rel18_tasks(...)` to
   install the Rel-18 handles *before* the task base is cloned, and finally
   `UeApp::spawn_tasks`.
4. **Spawn the actors.** `spawn_tasks` launches, in order: the **TUN** task
   (`TunTask::run`) plus a small TUN→app handler closure; the **App** task
   (`AppTask`, with a CLI server unless `--disable-cmd`); the **NAS** task
   (`UeApp::run_nas_task`); the **RRC** task (`UeApp::run_rrc_task`); the Rel-18
   **Ranging / MINT / Sidelink** tasks; any enabled **6G** tasks; and last the
   **RLS** task (`RlsTask::from_ue_config`). `-n > 1` spawns that many `UeApp`
   instances with incrementing IMSIs.

All tasks are async actors created with a default queue depth of
`DEFAULT_CHANNEL_CAPACITY = 256` (`tasks.rs`). Graceful shutdown is a
`tokio::sync::watch` broadcast driven by `TaskManager::shutdown` →
`UeTaskBase::shutdown_all`.

## Module map

The library surface is declared in `nextgsim-ue/src/lib.rs`. Each internal module
and the crate that backs it:

| Module (path under `nextgsim-ue/src/`) | Backing crate(s) | Responsibility |
|---|---|---|
| `main.rs` | — | `nr-ue` entry: CLI, config load/validate, task spawning, the NAS and RRC task loops (`run_nas_task`, `run_rrc_task`) and the MM/SM output pumps (`process_mm_outputs`, `process_sm_outputs`) |
| `tasks.rs` | `nextgsim-common` | Actor framework: `Task` trait, `TaskManager`, `UeTaskBase` (holds `app_tx/nas_tx/rrc_tx/rls_tx` `TaskHandle`s), the `TaskId` enum, and every inter-task `*Message` type |
| `app/` | `nextgsim-common` | App task, CLI command parsing/handling (`parse_ue_cli_command`, `cli_handler.rs`), status/`info` reporting (`status.rs`), config loading (`config_loader.rs`) |
| `nas/mm/` | `nextgsim-nas`, `nextgsim-crypto` | 5GMM: `MmOrchestrator`, MM state machine, SUCI build, authentication, NAS security, deregistration/service-request/config-update/emergency/MINT procedures |
| `nas/sm/` | `nextgsim-nas` | 5GSM: `SmOrchestrator`, PDU-session establishment/modification/release, PSI/PTI allocation, T3580/T3581/T3582, UL/DL NAS Transport wrapping |
| `rrc/` | `nextgsim-rrc`, `nextgsim-crypto` | RRC state machine, cell selection (TS 38.304), measurement, handover, RedCap, re-establishment, resume, AS security (`security.rs`), UAV RRC types |
| `rls/` | `nextgsim-rls`, `nextgsim-rlc` | Radio Link Simulation transport: cell search over UDP, RRC + user-plane PDU transport, per-PSI RLC (UM SN12) segmentation/reassembly |
| `tun/` | `nextgsim-common`, `tun-rs` | TUN device lifecycle and IP-packet I/O for the user plane (`TunTask`, `TunInterface`, `packet.rs`) |
| `timer.rs` | — | NAS timer constants + `NasTimerManager`, `GprsTimer2/3` decoders |
| `daps.rs` | — | Rel-17 DAPS handover UE state (`UeDapsState`) — dual-RLC handover model |
| `ambient_iot/`, `prose.rs`, `ranging/`, `sidelink/`, `uav.rs`, `mint/` | (in-crate) | Rel-17/18 5G-Advanced prototypes (see [6G & AI prototypes](#6g--ai-prototypes-pointers)) |
| `semantic_codec/`, `she_client/`, `isac_sensor/`, `fl_participant/`, `nwdaf_reporter/` | `nextgsim-semantic`, `nextgsim-she`, `nextgsim-isac`, `nextgsim-fl`, `nextgsim-nwdaf` | Feature-gated 6G/AI client tasks (see [6G & AI prototypes](#6g--ai-prototypes-pointers)) |

The composed crates (declared in `Cargo.toml`):

| Crate | What the UE uses it for |
|---|---|
| `nextgsim-common` | `UeConfig` + all config types, `OctetString`, `TaskMessage`, the CLI server, `Plmn`/`Tai`/`Supi` types |
| `nextgsim-nas` | 5GMM/5GSM message codecs, NAS headers, `security::{NasSecurityContext, compute_nas_mac, verify_nas_mac, nas_cipher, CipheringAlgorithm, IntegrityAlgorithm}` |
| `nextgsim-crypto` | Milenage, TUAK, NEA/NIA ciphers, ECIES SUCI concealment, and the 5G KDF hierarchy (`kdf::{derive_kausf, derive_kseaf, derive_kamf, derive_kgnb, derive_res_star}`) |
| `nextgsim-rrc` | ASN.1 UPER RRC codec (`codec::generated`) and RRC procedures (`procedures::security_mode`) |
| `nextgsim-rls` | The UDP RLS protocol: `UeCellSearch`, `RlsTransport`, `RrcChannel`, message codec |
| `nextgsim-rlc` | `RlcEntity` (TS 38.322) — UM SN12 segmentation/reassembly for the user plane |
| `nextgsim-gtp` | GTP-U codec (linked as a workspace dependency). On the matched-sim path the UE's user plane rides **RLS** to the gNB, and the **gNB** is the GTP-U (N3) endpoint toward the UPF — see [gNB Architecture](gnb.md) |

### Task topology

```text
                 ┌─────────────────────── UeTaskBase (clones of TaskHandles) ───────────────────────┐
                 │                                                                                   │
   CLI (nr-cli) ─┴─► App task ──► NAS task ──► RRC task ──► RLS task ══UDP══► gNB (nextgsim-gnb)
                        ▲            │  ▲          │            │  ▲
                        │           MM/SM      UL/DL RRC     per-PSI RLC
                        │        orchestrators  wrapping     (UM, SN12)
                        │            │                          │
   TUN device ◄─────────┴──── TunTask ◄── downlink IP    uplink IP ──► (to RLS)
```

Every arrow is an `mpsc` channel carrying a `TaskMessage<T>`; the message enums
(`AppMessage`, `NasMessage`, `RrcMessage`, `RlsMessage`, plus the Rel-18/6G
variants) are all defined in `tasks.rs`.

## NAS & security stack

The NAS layer is split into mobility management (`nas/mm/`) and session management
(`nas/sm/`) per `nas/mod.rs`. Both orchestrators are **sans-IO**: the NAS task in
`main.rs` (`run_nas_task`) feeds them downlink PDUs and a one-second `tick()`, and
they return output vectors (`MmOutput` / `SmOutput`) describing PDUs to send and
outcomes to act on. `process_mm_outputs` / `process_sm_outputs` in `main.rs`
translate those into `RrcMessage::UplinkNasDelivery` and TUN commands.

### The MM orchestrator and state machine

`MmOrchestrator` (`nas/mm/orchestrator.rs`) owns the MM procedure state, the NAS
timers, and the NAS security context. It is seeded from `MmUeIdentity::from_config`,
which precomputes: the SUPI string, the SUCI bytes, the subscriber key `k`, the
operator key `opc` (via `nextgsim_crypto::milenage::compute_opc` when `op_type`
is `Op`), the serving-network name, IMEISV, the advertised EA/IA capability
octets, the encoded Requested NSSAI, and the Rel-17/18 indications
(`snpn_nid`, `disaster_roaming`, `uav_indication`, `redcap`).

The state machine (`nas/mm/state.rs`, `MmStateMachine`) tracks four orthogonal
5GMM state axes per TS 24.501 §5.1.3 (per code comments):

| Axis | Type | Values |
|---|---|---|
| Registration | `RmState` | `Deregistered`, `Registered` |
| Connection | `CmState` | `Idle`, `Connected` |
| Main MM | `MmState` | `Null`, `Deregistered`, `RegisteredInitiated`, `Registered`, `DeregisteredInitiated`, `ServiceRequestInitiated` |
| Sub-state | `MmSubState` | e.g. `DeregisteredNormalService`, `RegisteredNormalService`, `RegisteredPlmnSearch`, … |
| Update status | `UpdateStatus` | `Updated`, `NotUpdated`, `RoamingNotAllowed` |

`switch_mm_state` derives the `RmState` from the target `MmState`
(`derive_rm_state`) so RM and MM never diverge.

`MmOutput` (`orchestrator.rs`) is the orchestrator's vocabulary to the NAS task:
`SendNasPdu`, `RegistrationSucceeded`, `EquivalentPlmnsUpdated`,
`RegistrationFailed(MmCause)`, `AuthenticationRejected`, `PlmnSearchNeeded`,
`NotHandled` (a plain 5GMM PDU the orchestrator does not own — e.g. a 5GSM
piggyback, dispatched onward), and `AsSecurityKgnb` (only when the AS-security
gate is on).

### SUCI construction

`build_suci` (`nas/mm/suci.rs`) assembles the 5GS mobile-identity IE per
TS 24.501 §9.11.3.4 (per code comments): SUPI-format/type octet, home-network
PLMN (TBCD), routing indicator, protection-scheme id, key id, and the scheme
output. Three schemes are implemented:

- **Null (scheme 0):** cleartext MSIN in TBCD.
- **Profile A (scheme 1):** ECIES over X25519 via
  `nextgsim_crypto::ecies::generate_suci_profile_a`.
- **Profile B (scheme 2):** ECIES over secp256r1 via `generate_suci_profile_b`.

A **fresh ephemeral key pair is generated per call** for the ECIES profiles, and
there is deliberately **no fall-back to the null scheme** — a UE configured for
ECIES that cannot conceal returns `SuciBuildError` rather than leaking the SUPI
(source comment cites TS 33.501 §6.12). The module's unit tests assert byte-exact
output, including deconcealment against the TS 33.501 **Annex C.4.3 (Profile A)**
and **Annex C.4.4 (Profile B)** home-network key pairs and a tampered-MAC
rejection.

### Authentication and the key hierarchy

When an `AuthenticationRequest` arrives, `MmOrchestrator`
(`orchestrator.rs`, `handle_authentication_request`) decodes it
(`AuthenticationRequest::decode`) and runs Milenage:
`Milenage::new(&self.identity.k, &self.identity.opc)` produces the vector, then
the 5G key hierarchy is derived with `nextgsim-crypto`'s KDFs (TS 33.501 Annex A,
per code comments):

```text
K ──Milenage──► CK, IK, RES, AK, MAC
                 │
   derive_res_star(ck, ik, sn_name, rand, res) ─► RES*  (Auth Response)
   derive_kausf(ck, ik, sn_name, autn)          ─► KAUSF
   derive_kseaf(kausf, sn_name)                 ─► KSEAF
   derive_kamf(kseaf, supi, abba)               ─► KAMF
                 │
   nextgsim_nas::security  ─► Knas_int, Knas_enc  (NAS integrity + ciphering)
   derive_kgnb(kamf, ul_count, 0x01)            ─► KgNB (AS security, gated)
```

NAS message protection is applied by `MmOrchestrator::protect_if_active`: once the
security context is active, uplink PDUs are integrity-protected and ciphered
(`nextgsim_nas::security::{compute_nas_mac, nas_cipher}`) before they leave as
`MmOutput::SendNasPdu`; downlink MAC verification and deciphering happen in
`handle_downlink`. The abnormal authentication cases — MAC failure (cause #20),
non-5G network / separation-bit (#26), and SQN sync failure with a real Milenage
f1\*/f5\* AUTS — are handled in the orchestrator (source comments cite TS 24.501
§5.4.1.3.6).

**What is real vs. what is available.** The UE's **live** authentication path
supports both **5G-AKA and EAP-AKA′**, both of which are **Milenage-based**.
`orchestrator.rs` imports `milenage::{compute_opc, Milenage}` for the 5G-AKA
branch, and `handle_authentication_request` dispatches to
`handle_eap_aka_prime_challenge` when an `AuthenticationRequest` carries an EAP
message instead of RAND/AUTN; that handler calls
`nextgsim_crypto::eap_aka_prime::run_eap_aka_prime` and derives the
KAUSF/KSEAF/KAMF hierarchy. The `nextgsim-crypto` crate *also* ships **TUAK**,
whose unit tests are **validated byte-exact against the TS 35.232 Implementers'
Test Data sets** (source comment in `nextgsim-crypto/src/tuak.rs`: "Vectors:
validated byte-exact against TS 35.232 Implementers' Test Data sets"), and
Milenage is checked against the TS 35.207 test sets — but **TUAK** is not wired
into the UE's live orchestrator.

### The SM orchestrator

`SmOrchestrator` (`nas/sm/orchestrator.rs`) owns per-PSI session state, PSI (1–15)
and PTI allocation, and the T3580/T3581/T3582 timers. `SmSessionParams::from_config`
turns the configured `sessions` into request parameters (defaulting to a single
IPv4 `internet` session on the first configured slice). `start_establishment`
builds a `PduSessionEstablishmentRequest` and wraps it in a `UlNasTransport`
(`PayloadContainerType::N1SmInformation`) per TS 24.501 §5.4.5 (per code comments);
the plain PDU is then protected by the **MM** security context in
`process_sm_outputs` (`orch.protect_if_active(plain)`) before transmission —
5GSM never carries its own security. The XR-DNN→5QI mapping (Rel-18) and the MINT
subscription routing are applied here too.

### Driving registration from the UE side

The end-to-end trigger chain, all in `main.rs` unless noted:

1. The **RRC task** selects a serving cell on the first `RrcMessage::SignalChanged`
   and sends `NasMessage::PerformMmCycle` to the NAS task
   (`run_rrc_task`, `registration_triggered` latch).
2. `PerformMmCycle` in a `Deregistered` UE calls
   `orch.start_registration(RegistrationType::InitialRegistration)`.
3. The resulting `MmOutput::SendNasPdu` becomes `RrcMessage::UplinkNasDelivery`
   → RRC → RLS → gNB.
4. Downlink NAS (`NasMessage::NasDelivery`, EPD `0x7E`) is fed to
   `orch.handle_downlink`, which walks Authentication → Security Mode Command →
   Registration Accept.
5. On `MmOutput::RegistrationSucceeded`, `process_mm_outputs` establishes the
   configured default PDU sessions via `sm_orch.establish_default_sessions`.

The message-level detail of the network side of this exchange is in the
[registration flow](../concepts/registration-flow.md) concept chapter.

## RRC & radio-link

The RRC module (`rrc/mod.rs`) re-exports a broad procedure surface —
`cell_selection`, `handover`, `measurement`, `redcap`, `reestablishment`,
`resume`, `security`, `state`, `uav`. The live state machine is
`RrcStateMachine` (`rrc/state.rs`): `Idle` / `Connected` / `Inactive` driven by
`RrcStateTransition`.

The RRC task loop (`run_rrc_task` in `main.rs`) is deliberately thin:

- **Cell selection.** It tracks `discovered_cells: HashMap<cell_id, dbm>` and, on
  the first `SignalChanged` while `Idle`, picks the strongest cell
  (`max_by_key` on dBm), sends `RlsMessage::AssignCurrentCell`, and triggers NAS
  registration.
- **Uplink NAS.** `UplinkNasDelivery` is wrapped in a simplified **UL Information
  Transfer** frame (`[0x08, 0x00, …NAS]`) once connected, or sent as a **raw NAS
  PDU** for initial access (the gNB auto-creates the UE context) — see the byte
  prefixes in `run_rrc_task`.
- **Downlink RRC.** A **DL Information Transfer** frame (leading `0x04`) is
  unwrapped and the inner NAS PDU is forwarded to the NAS task as
  `NasMessage::NasDelivery`.

**AS security is behind a wire gate.** `rrc/security.rs` defines
`pub const I5_UE_AS_SECURITY: bool = false;` — **default-off**. When false, the
DL-DCCH handling is byte-for-byte the legacy path so the matched-sim E2E is
unchanged. When flipped on, the RRC task decodes the AS `SecurityModeCommand`
(`nextgsim_rrc::procedures::security_mode::decode_security_mode_command`), derives
the AS keys (`AsSecurityContext::derive_from_kgnb`, using the `KgNB` handed down
from the NAS plane via `RrcMessage::AsSecurityKey`), and replies with
`SecurityModeComplete`. The derivation is **fail-closed** (no KgNB/algorithm →
not activated) and explicitly refuses NEA3 (unsupported keystream). `security.rs`
also implements the ShortMAC-I (TS 38.331 §5.3.7.4) and resumeMAC-I (§5.3.13.3)
computations over the UPER-encoded `VarShortMAC_Input` / `VarResumeMAC_Input`
ASN.1 types from `nextgsim-rrc`, using the NIA algorithms from `nextgsim-crypto`.

Below RRC, the **RLS task** (`rls/task.rs`, `RlsTask::from_ue_config`) is the
simulated radio. It binds a UDP socket, runs `UeCellSearch` (heartbeats every
~1000 ms to every address in `gnb_search_list`, `DEFAULT_RLS_PORT = 4997`), and
carries RRC and user-plane PDUs to the serving gNB through `RlsTransport`. There
is no real PHY/MAC: RLS *is* the wire between UE and gNB.

## User-plane / TUN

Once a PDU session is active, the UE terminates its user plane on a TUN device.
The lifecycle and packet paths (`main.rs`, `tun/task.rs`, `rls/task.rs`):

**Interface lifecycle.** `SmOutput::SessionEstablished { psi, ipv4 }` in
`process_sm_outputs` sends `TunMessage::CreateInterface`, and `SessionReleased`
sends `TunMessage::DestroyInterface`. `TunTask` (`tun/task.rs`) creates a
`TunInterface` (a `tun-rs` `AsyncDevice`) named from `TunTaskConfig::default()`
(prefix `uesimtun`, MTU 1400), keyed by PSI.

**Uplink (TUN → gNB):**

1. A reader task on the TUN device emits `TunAppMessage::UplinkData { psi, data }`.
2. The handler in `spawn_tasks` (`main.rs`) does two things: sends
   `NasMessage::InitiateServiceRequest` (so a CM-IDLE UE wakes and runs Service
   Request), and forwards the packet as `RlsMessage::DataPduDelivery { psi, pdu }`
   straight to the RLS task.
3. `RlsTask::handle_data_pdu_delivery` submits the SDU to the **per-PSI RLC
   entity** — created as `RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12)`
   (`rlc_entity_for`) — segments it with `build_pdu(1500)`, and sends each RLC PDU
   as an `RlsProtocolMessage::PduTransmission` over UDP to the serving gNB.

**Downlink (gNB → TUN):**

1. `RlsTask` receives a `PduTransmission`; `RlsTransport::process_pdu_transmission`
   yields a `TransportEvent::DataReceived { psi, data }`.
2. The RLC entity reassembles (`receive_pdu` + `poll_reassembled`), and each SDU
   is sent to the NAS task as `NasMessage::UplinkDataDelivery { psi, data }`.
3. The NAS task forwards it to the TUN task as `TunMessage::WriteData { psi, data }`,
   which writes the IP packet to the device.

Note the naming quirk: the `NasMessage::UplinkDataDelivery` variant actually
carries **downlink** data toward the TUN — its handler comment in `run_nas_task`
says so explicitly ("This is actually downlink data (from network to UE)").

On the matched-sim path the gNB is the GTP-U (N3) endpoint to the UPF; the UE's
GTP-U tunnel is what the E2E ping traverses. The full data-plane trace across UE,
gNB, and UPF is in [PDU session & user plane](../concepts/pdu-session-userplane.md).

## 6G & AI prototypes (pointers)

These modules are **non-normative research prototypes**. The Rel-18 tasks
(Ranging, MINT, Sidelink) are always compiled and spawned; the 6G/AI client tasks
are both Cargo-feature-gated (`nextgsim-she`, `nextgsim-nwdaf`, `nextgsim-isac`,
`nextgsim-fl`, `nextgsim-semantic`) **and** config-gated
(`she_enabled`, `ai_ml_enabled`, `isac_enabled`, `federated_learning_enabled`,
`semantic_comm_enabled`) in `spawn_tasks`. For what they model and their status,
see [The 6G / AI stack](../concepts/ai-6g-stack.md).

| Module (`nextgsim-ue/src/…`) | One-line honest pointer |
|---|---|
| `ambient_iot/` | Rel-18 Ambient IoT fleet management model (TS 22.369) — device-group coordination, no wire protocol. |
| `prose.rs` | ProSe PC5 proximity-services model (TS 23.303/23.304) — discovery + UE-to-UE relay state, prototype. |
| `ranging/` | Rel-18 UE-to-UE ranging / carrier-phase positioning (TS 23.586) — RTT/phase model + LMF report, prototype. |
| `sidelink/` | NR sidelink relay, PC5 link, sidelink positioning — prototype state machines. |
| `uav.rs` | Rel-17/18 UAV context (TS 23.256) — wraps the RRC UAV types with NAS authorization/C2/geofence modelling. |
| `mint/` | Rel-18 MINT / multi-USIM (TS 23.761) — multi-SUPI secondary-subscription driver, integrated with the NAS task. |
| `semantic_codec/` | 6G semantic-communication encode/decode task — research prototype. |
| `she_client/` | 6G Service-Hosting-Environment edge-inference/offload client — research prototype. |
| `isac_sensor/` | 6G Integrated Sensing & Communication sensor task — research prototype. |
| `fl_participant/` | 6G Federated-Learning local-training participant — research prototype. |
| `nwdaf_reporter/` | 6G NWDAF measurement reporter / analytics client — research prototype. |

## Simplifications & known gaps

Grounded in the code, these are the honest deviations from a textbook UE:

- **RLS is the radio; there is no PHY/MAC.** UE↔gNB is UDP RLS
  (`rls/task.rs`); the only Layer-2 processing on the user plane is RLC in
  **UM with SN12** (`rlc_entity_for`) — no AM, no HARQ, no scheduler.
- **AS security is default-off.** `I5_UE_AS_SECURITY = false`
  (`rrc/security.rs`): on the matched-sim path SRBs are **not** integrity-protected
  or ciphered at the AS layer. NAS-layer security (Knas_int/Knas_enc) *is* real
  and active. Flipping the gate on requires the docker A/B sign-off noted in the
  source (it mirrors the gNB's `C5_TYPED_DCCH_DISPATCH` gate).
- **5G-AKA and EAP-AKA′ on the live path; TUAK unwired.** The live orchestrator
  handles both **5G-AKA** and **EAP-AKA′**, which are both Milenage-based
  (`handle_eap_aka_prime_challenge` calls
  `nextgsim_crypto::eap_aka_prime::run_eap_aka_prime`). **TUAK** also exists in
  `nextgsim-crypto` (byte-exact vs TS 35.232 in its own tests) but is not wired
  into the UE orchestrator.
- **RRC framing is simplified.** Uplink/downlink NAS is carried with fixed
  one/two/three-byte UL/DL Information Transfer prefixes (`0x08` / `0x04`) and a
  raw-NAS fallback for initial access, not full ASN.1 RRC containers
  (`run_rrc_task`).
- **UAV tracking uses a sim-private message type.** The UE emits a UAV tracking
  report on 5GMM message type `0x6A` (`UAV_TRACKING_REPORT_MSG_TYPE`,
  `nas/mm/orchestrator.rs`); the source comment is explicit that **no such
  UE-originated UAV NAS message exists in 3GPP** — it is a self-contained geofence
  demo, kept in sync with the matched core.
- **6G RRC messages are logged and dropped in the binary.** The `Sixg*` arms of
  `RrcMessage` in `run_nas_task`/`run_rrc_task` are handled only by the library's
  `RrcTask`; in the `nr-ue` binary loop they are debug-logged and discarded.
- **Cooperative peer.** Everything above is exercised against the matched
  nextgsim gNB and NextGCore core, not an arbitrary commercial network.

## Where to look next

- [UE Configuration](../configuration.md#ue-configuration) — every field that
  seeds `UeConfig` (SUPI, keys, `protection_scheme`, `sessions`, `gnb_search_list`,
  `tun_name`, and the Rel-17/18 feature blocks).
- [nr-cli Reference](../cli-reference.md) — the `info`/`status`/`ps-establish`/
  `deregister` commands that drive the App and NAS tasks at runtime.
- [UE Registration Call Flow](../concepts/registration-flow.md) — the message-by-message
  registration exchange this UE participates in.
- [PDU Session & User Plane](../concepts/pdu-session-userplane.md) — the end-to-end
  data-plane trace (TUN ↔ RLS ↔ gNB ↔ GTP-U ↔ UPF).
- [The 6G / AI Stack](../concepts/ai-6g-stack.md) — the prototype tasks summarised
  above.
- [gNB Architecture](gnb.md) — the peer that terminates RLS and owns GTP-U/N3.
