# UE Registration and the RRC/NGAP/NAS Stack

This chapter traces 5G **initial registration** as the nextgsim simulator
*actually* runs it — from the UE powering on and discovering a cell, through the
gNB relaying the initial NAS message to the AMF, 5G-AKA authentication, NAS and
AS security setup, and the piggybacked Registration Accept. It follows the code
on **both simulator sides** (`nextgsim-ue` and `nextgsim-gnb`) and names the
crate, file, and function for every step, so you can open the source and read
along. The far side of the N2 association is the separate **nextgcore** core
network (AMF/AUSF/UDM); here it appears only as the cooperative peer the
simulator drives.

The chapter assumes the layered picture from [gNB architecture](../architecture/gnb.md)
and [UE architecture](../architecture/ue.md), and the knobs from the
[Configuration Reference](../configuration.md). The PDU-session/user-plane leg
that follows registration is covered in
[PDU Sessions and the User Plane](pdu-session-userplane.md).

> **Honesty note:** This flow is validated by nextgsim's own per-crate unit
> tests, in-process strict-peer integration tests, and a **matched-simulator**
> docker E2E in which the project drives *its own* gNB/UE against nextgcore
> (UE→UPF GTP-U data plane verified end-to-end) — it is **not** third-party
> conformance certified, and the peer is cooperative rather than an arbitrary
> commercial core. The live registration path runs **5G-AKA only**. An EAP-AKA′
> path exists in `nextgsim-ue/src/nas/mm/orchestrator.rs`
> (`handle_eap_aka_prime_challenge`) but is entered *only* if the Authentication
> Request carries an EAP-message IE instead of RAND/AUTN; the matched core sends
> RAND/AUTN, so EAP-AKA′ is not exercised on the happy path. 3GPP TS numbers are
> quoted only where they appear in this repo's source comments, phrased "per
> code comments". The 6G/AI feature crates referenced elsewhere are
> non-normative research prototypes (no frozen Rel-20 Stage-3 exists).

## Components involved

Registration crosses six nextgsim crates plus the crypto library. The "radio"
is simulated: RLS carries RRC and user-plane PDUs over UDP, and NGAP rides SCTP
(userspace `sctp-proto` matched to nextgcore, or kernel SCTP).

| Layer | Crate / module | Role in registration |
|---|---|---|
| RLS (sim radio) | `nextgsim-rls`; `nextgsim-ue/src/rls/task.rs`, `nextgsim-gnb/src/rls/task.rs` | Heartbeat cell discovery, dBm estimation, RRC/data PDU transport over UDP:4997 |
| Cell selection | `nextgsim-ue/src/rrc/cell_selection.rs` | TS 38.304-style suitable/acceptable cell selection; TS 23.122 PLMN selection |
| RRC | `nextgsim-rrc` (procedures); `nextgsim-ue/src/rrc/task.rs`, `nextgsim-gnb/src/rrc/{task.rs,connection.rs}` | RRCSetupRequest / RRCSetup / RRCSetupComplete; SRB1; carries initial NAS |
| NGAP | `nextgsim-ngap` (procedures); `nextgsim-gnb/src/ngap/task.rs` | InitialUEMessage, Downlink/UplinkNASTransport, InitialContextSetup |
| NAS 5GMM | `nextgsim-nas` (codec); `nextgsim-ue/src/nas/mm/orchestrator.rs` | Registration Request build, SUCI, 5G-AKA, NAS security, 5GMM state machine |
| Crypto | `nextgsim-crypto` | MILENAGE `f1..f5`, key hierarchy KDFs, ECIES SUCI, NAS ciphering |

Cross-links: the far-side handlers (AMF `handle_initial_ue_message`,
`start_authentication`, `send_registration_accept`) live in nextgcore and are
described in its own *UE Registration Call Flow* chapter — treat them as the
peer here.

## The transport underneath

There is **no real radio**. The RLS protocol (`nextgsim-rls/src/protocol.rs`,
version 3.2.7) defines four message types — `Heartbeat` (4), `HeartbeatAck` (5),
`PduTransmission` (6), `PduTransmissionAck` (7) — sent as UDP datagrams on port
`4997` (`DEFAULT_RLS_PORT`). RRC and user-plane bytes travel inside
`PduTransmission` with a `PduType` of `Rrc` (1) or `Data` (2). Only user-plane
`Data` PDUs are run through an RLC UM SN12 entity (`nextgsim-rlc`) for
segmentation/reassembly — one entity per PDU Session ID (`rlc_entities`, keyed by
PSI in `nextgsim-ue/src/rls/task.rs`, created on first use by `rlc_entity_for`).
**RRC PDUs bypass RLC entirely.** On receive, `handle_pdu_transmission` routes
`TransportEvent::RrcReceived` straight to RRC
(`RrcMessage::DownlinkRrcDelivery`) with no RLC, while only
`TransportEvent::DataReceived { psi, .. }` is fed through
`rlc_entity_for(psi).receive_pdu(..)` for reassembly. On send, only
`handle_data_pdu_delivery(psi, ..)` submits SDUs to RLC
(`rlc_entity_for(psi).submit_sdu(..)` / `build_pdu(1500)`); the RRC transmit path
(`handle_rrc_pdu_delivery`) never calls `rlc_entity_for`.

NGAP rides SCTP to the AMF via `nextgsim-gnb/src/ngap/task.rs`; the AMF endpoint
comes from `amf_configs` (default port `38412`). Everything from
InitialUEMessage onward is UE-associated signalling over an SCTP stream.

## Step-by-step flow

The whole procedure as one ladder (initial registration, SUCI identity, matched
peer):

```text
 UE (nextgsim-ue)        gNB (nextgsim-gnb)              AMF (nextgcore)
   |                        |                                |
 1 |  Heartbeat (UDP 4997)  |                                |
   |----------------------->| GnbCellTracker.process_hb      |
 2 |  HeartbeatAck (dBm)    |                                |
   |<-----------------------|                                |
 3 | cell select + fabricate SIB1                            |
   | (CellSelector)         |                                |
 4 | RRCSetupRequest (UL-CCCH, random UE id)                 |
   |----------------------->| process_rrc_setup_request      |
 5 | RRCSetup (DL-CCCH, SRB1)|                               |
   |<-----------------------|                                |
 6 | RRCSetupComplete (UL-DCCH) + [Registration Request NAS] |
   |----------------------->| finish_rrc_setup_complete      |
 7 |                        | InitialUEMessage (NGAP/SCTP)   |
   |                        |------------------------------->| handle_initial_ue_message
 8 |         Authentication Request (DL NAS Transport)       |
   |<-----------------------|<-------------------------------| start_authentication
 9 |         Authentication Response (RES*) (UL NAS)         |
   |----------------------->|------------------------------->| verify RES*
10 |         NAS Security Mode Command (DL NAS)              |
   |<-----------------------|<-------------------------------|
11 | Security Mode Complete (+ replayed Reg Request) (UL NAS)|
   |----------------------->|------------------------------->|
12 |     InitialContextSetupRequest {KgNB, caps, [Reg Accept]}
   |                        |<-------------------------------| send_registration_accept
   |  RRC SecurityModeCommand (SRB1)  +  Registration Accept |
   |<-----------------------| activate_as_security           |
13 |                        | InitialContextSetupResponse    |
   |                        |------------------------------->|
14 | Registration Complete (if GUTI assigned) (UL NAS)       |
   |----------------------->|------------------------------->|  REGISTERED
```

### 1–2. RLS cell discovery (heartbeat)

The UE's `RlsTask` (`nextgsim-ue/src/rls/task.rs`) fires
`send_heartbeats` on a 1 s tokio interval (`HEARTBEAT_INTERVAL_MS = 1000`),
building one `RlsHeartbeat` per address in `gnb_search_list` via
`UeCellSearch::create_heartbeats` (`nextgsim-rls/src/cell_search.rs`). Each
heartbeat carries the UE's random `sti` (Simulated Transmission Identifier, set
in `RlsTask::new`) and a `Vector3` simulated position. Search-list entries with
no port default to `4997`.

The gNB's `RlsTask` (`nextgsim-gnb/src/rls/task.rs`) receives it in
`handle_heartbeat`, which calls `GnbCellTracker::process_heartbeat`. Signal
strength is *modelled from geometry*: `estimate_dbm` returns `-distance` (the
Euclidean distance between the gNB's `phy_location` and the heartbeat's
`sim_pos`); anything weaker than `MIN_ALLOWED_DBM = -120` is dropped. A new STI
allocates a UE id and emits `UeDetected` (→ `RrcMessage::SignalDetected`), then
the gNB replies `RlsHeartbeatAck::with_dbm`. The gNB's STI is its configured
`nci` (`RlsTask::new`).

Back on the UE, `handle_heartbeat_ack` → `UeCellSearch::process_heartbeat_ack`
raises `CellDiscovered { cell_id, dbm }` and the RLS task forwards
`RrcMessage::SignalChanged` to the RRC task. Loss detection is by timeout: a
cell unseen for `heartbeat_threshold` (2000 ms) becomes `CellLost`, which — if
it was the serving cell — raises a `RadioLinkFailure`.

### 3. Cell selection and fabricated system information

The UE RRC task (`nextgsim-ue/src/rrc/task.rs`) handles `SignalChanged` in
`handle_signal_changed` → `CellSelector::handle_signal_change`
(`nextgsim-ue/src/rrc/cell_selection.rs`). On a newly detected cell it calls
`provide_simulated_system_info`, which **fabricates the MIB and SIB1 locally**
(not barred, not reserved, `q_rx_lev_min = -70`, PLMN = the UE's configured
HPLMN, and the SNPN NID only when `snpn_config` is set). This is a deliberate
simplification: there is no on-air SIB broadcast, so the UE synthesises just
enough system information to make the discovered cell selectable.

`perform_cell_selection` runs `CellSelector::perform_cell_selection` (cited
against TS 38.304 §5.2 per code comments): after a 1 s startup delay it looks
for a *suitable* cell (SIB1 present, PLMN matches the NAS-selected PLMN, not
barred/reserved, TAI not forbidden, `meets_s_criteria` i.e. `Srxlev > 0`), else
an *acceptable* cell (any PLMN). The winner (highest `ranking_value`) is sent to
RLS as `RlsMessage::AssignCurrentCell` and to NAS as
`NasMessage::ActiveCellChanged`. PLMN selection itself is the TS 23.122 §4.4.3
`PlmnSelector` in the same file; in the sim every cell broadcasts the home PLMN,
so selection resolves to the HPLMN (`nextgsim-ue/src/main.rs`).

### 4. Registration Request build (NAS) and RRC establishment kickoff

Registration is what *triggers* the RRC connection, not the reverse. After PLMN
selection succeeds, `nextgsim-ue/src/main.rs` calls
`orch.start_registration(RegistrationType::InitialRegistration)` on the MM
orchestrator (`nextgsim-ue/src/nas/mm/orchestrator.rs`).

`build_and_send_registration` assembles a `RegistrationRequest`
(`nextgsim-nas`): `ngKSI` = `NasKeySetIdentifier::no_key()` (no security yet),
mobile identity = the **SUCI** (`Ie5gsMobileIdentity` of type `Suci`) unless a
valid 5G-GUTI is stored, UE security capability `[ea_cap, ia_cap]`, requested
NSSAI, and last-visited TAI. The SUCI itself is built by `build_suci`
(`nextgsim-ue/src/nas/mm/suci.rs`) per TS 24.501 §9.11.3.4: the null scheme
emits the cleartext MSIN in TBCD, while Profile A (X25519) and Profile B
(secp256r1) conceal the MSIN via ECIES (`nextgsim-crypto/src/ecies.rs`) — the
byte layout is golden-tested against TS 33.501 Annex C key pairs.

The **cleartext-IE gate** (per code comments TS 24.501 §4.4.6) is enforced by
the UE, not just the core: because this initial request is *unprotected*, only
cleartext IEs go on the wire (UE security capability, and the SNPN NID / disaster
indication when applicable). Requested NSSAI and last-visited TAI are stripped
from the clear message but the **full** request is stashed in
`self.last_registration_request` to be replayed inside Security Mode Complete
(step 11). `T3510` is armed and the state moves to `RegisteredInitiated`. The
resulting NAS PDU is emitted as `MmOutput::SendNasPdu` and handed to RRC as
`RrcMessage::UplinkNasDelivery`.

The UE RRC task's `handle_uplink_nas_delivery` sees the RRC state is `Idle`,
stores the PDU as `initial_nas_pdu`, and calls `start_connection_establishment`:
it encodes an `RRCSetupRequest` via `encode_rrc_setup_request`
(`nextgsim-rrc/src/procedures/rrc_setup.rs`) with a random 39-bit UE identity
(`UeIdentity::RandomValue`) and establishment cause 3, which the encoder maps to
`mo-Signalling`. It is sent on **UL-CCCH** over RLS.

### 5. gNB answers RRCSetup (SRB1)

The gNB RRC task (`nextgsim-gnb/src/rrc/task.rs`) receives UL-CCCH in
`handle_ul_ccch_message`, decodes the request with `decode_rrc_setup_request`,
and calls `RrcConnectionManager::process_rrc_setup_request`
(`nextgsim-gnb/src/rrc/connection.rs`). That creates the gNB-side UE context,
allocates the RRC transaction id from the per-UE allocator (deterministically
`0` on a fresh context), and builds the **RRCSetup** via `build_rrc_setup` →
`srb1_rrc_setup_params` + `encode_rrc_setup`. Per code comments TS 38.331
§5.3.5.6.3 this PDU establishes **SRB1** (a `RadioBearerConfig` with
SRB-Identity 1 plus a masterCellGroup carrying one RLC-BearerConfig at LCID 1);
the emitted bytes are pinned to a hand-derived golden UPER PDU
(`[0x20,0x40,0x00,0x22,0x00,0x04,0x00,0x00]`, verified in
`test_rrc_setup_request_emits_golden_srb1_pdu_tid0`). It is returned on
**DL-CCCH**. The cell must be unbarred first (`RadioPowerOn` →
`set_barred(false)`), otherwise the request is rejected.

The UE handles DL-CCCH message type `0x00` in `handle_dl_ccch_message` →
`handle_rrc_setup`: it decodes the payload with `decode_rrc_setup` and records
SRB1 in `apply_rrc_setup_config` (a decode failure is *tolerated* with a warning,
keeping the matched path default-safe), transitions the RRC state machine to
`Connected`, and calls `send_rrc_setup_complete`.

### 6. RRCSetupComplete carries the initial NAS

`send_rrc_setup_complete` (UE RRC task) takes the stored `initial_nas_pdu` and
encodes an **RRCSetupComplete** via `encode_rrc_setup_complete` with
`dedicated_nas_message = <Registration Request>`, `selected_plmn_identity = 1`,
and the echoed transaction id pinned to `0`. The RedCap indication rides here
when `config.redcap` is set. It is sent on **UL-DCCH** (SRB1).

The gNB's `handle_ul_dcch_message` tries an **ASN.1-first** decode with
`decode_rrc_setup_complete`; on success it routes to
`handle_rrc_setup_complete_asn1` → `finish_rrc_setup_complete`. (This ASN.1-first
path is the fix pinned by `tests/src/rrc_handshake.rs`: the UE's real UPER
RRCSetupComplete has leading byte `0x10`, which matched *no* arm of the legacy
`bytes[0] & 0x0F` nibble dispatcher and used to drop the registration NAS.) The
bespoke fallback framing `[0x04, tid, 0x01, NAS…]` is still accepted by
`handle_rrc_setup_complete`. `finish_rrc_setup_complete` calls
`RrcConnectionManager::process_rrc_setup_complete` (verifies the echoed tid,
marks the context Connected) and then `send_initial_nas_delivery`, and separately
sends a `UECapabilityEnquiry`.

### 7. gNB → AMF: InitialUEMessage (NGAP)

`send_initial_nas_delivery` posts `NgapMessage::InitialNasDelivery` to the NGAP
task, handled by `handle_initial_nas_delivery`
(`nextgsim-gnb/src/ngap/task.rs`). It selects an AMF (`select_amf`), creates the
NGAP UE context (allocating the RAN-UE-NGAP-ID), and fills
`InitialUeMessageParams` (`nextgsim-ngap/src/procedures/initial_ue_message.rs`):
the NAS PDU, `UserLocationInfoNr` (NR-CGI from `config.nci`, TAI from
`config.plmn`/`config.tac`), the RRC establishment cause, and
`UeContextRequest = Requested`. `encode_initial_ue_message` (cited against
TS 38.413 §8.6.1 per code comments) produces the APER PDU, sent UE-associated
over the SCTP stream. From here the AMF (nextgcore) drives the SBI side; the
simulator is a relay.

### 8–9. Authentication (5G-AKA)

The AMF's Authentication Request comes back as an NGAP **DownlinkNASTransport**,
handled by `handle_downlink_nas_transport` (`nextgsim-gnb/src/ngap/task.rs`),
which forwards the inner NAS PDU to RRC as `RrcMessage::NasDelivery`; the gNB RRC
task wraps it in a DL Information Transfer (`build_dl_information_transfer`,
DL-DCCH) and RLS carries it to the UE.

The UE NAS orchestrator dispatches every downlink NAS PDU through
`handle_downlink` (`orchestrator.rs`), which reads the security-header type and,
because security is not yet active, routes the plain message via `handle_plain`.
An `AuthenticationRequest` (allowed unprotected by the TS 24.501 §4.4.4.2
`allowed_when_plain` allow-list) reaches `handle_authentication_request`:

```text
AUTN = (SQN ⊕ AK) ‖ AMF ‖ MAC          (from the Authentication Request)
AK      = MILENAGE.f5(RAND)            → recover SQN
MAC?    = MILENAGE.f1(RAND, SQN, AMF)  → compare (mismatch ⇒ cause #20)
sep bit : AMF[0] & 0x80 must be 1      (else cause #26)
SQN?    : sqn_in_range()               (else cause #21 + AUTS resync)
RES     = MILENAGE.f2(RAND)
CK      = MILENAGE.f3(RAND)
IK      = MILENAGE.f4(RAND)
RES*    = derive_res_star(CK, IK, SN-name, RAND, RES)
```

MILENAGE lives in `nextgsim-crypto/src/milenage.rs` (`Milenage::new(k, opc)`,
`f1..f5`, plus `f1_star`/`f5_star` for the AUTS resync parameter computed by
`compute_auts`). The key hierarchy is derived immediately —
`derive_kausf` → `derive_kseaf` → `derive_kamf`
(`nextgsim-crypto/src/kdf.rs`), folding in the network-signalled ABBA and the
UE's SUPI — and stored in the partial NAS security context. The UE replies with
an `AuthenticationResponse` carrying `RES*`; the gNB relays it to the AMF as an
**UplinkNASTransport** (`handle_uplink_nas_delivery`). The AMF/AUSF verify
`RES*`. (TUAK is implemented in `nextgsim-crypto/src/tuak.rs` but the
registration path calls MILENAGE.)

### 10–11. NAS Security Mode Command / Complete

See [Security establishment](#security-establishment) below — this is where the
NAS security context is activated and the initial Registration Request is
replayed.

### 12. InitialContextSetupRequest with the piggybacked Registration Accept

Once NAS security is up and the AMF has run its registration SBI calls, it sends
an **InitialContextSetupRequest** with the Registration Accept piggybacked as the
NAS-PDU, plus the UE Security Capabilities and the **Security Key (KgNB)**. The
gNB handles it in `handle_initial_context_setup_request`
(`nextgsim-gnb/src/ngap/task.rs`), parsed from
`nextgsim-ngap/src/procedures/initial_context_setup.rs`
(`InitialContextSetupRequestData`). It calls `activate_as_security` (AS keys, see
below), then forwards the piggybacked Registration Accept to the UE via
`RrcMessage::NasDelivery`, transitions the UE context to active, sets up any PDU
sessions carried in the ICS Request (`PDUSessionResourceSetupListCxtReq`), and
replies **InitialContextSetupResponse** (`encode_initial_context_setup_response`).

The UE NAS orchestrator receives the (now integrity-protected and ciphered)
Registration Accept, decrypts/verifies it in `handle_downlink`, and dispatches to
`handle_registration_accept`: it stops `T3510`, moves to
`RegisteredNormalService`, and stores the assigned 5G-GUTI, TAI list, Allowed
NSSAI, equivalent PLMNs, and `T3512`.

### 13–14. Registration Complete → REGISTERED

If the Accept assigned a new 5G-GUTI (or a pending NSSAI / slicing-change / DRX
parameters — the `needs_ack` condition), `handle_registration_accept` emits a
**Registration Complete**, protected with the active context and relayed to the
AMF as an UplinkNASTransport. The UE is now registered; the gNB's
InitialContextSetupResponse (step 13) confirms the RAN context. Basic
registration is covered by `tests/src/ue_registration.rs`
(`test_ue_registration_basic`).

## Security establishment

Registration activates **two** security contexts in sequence: NAS security
(mandatory, fully live) and AS security (KgNB-derived; see the honesty caveat).

### NAS Security Mode Command → Complete

The AMF's Security Mode Command arrives as a downlink NAS message with a
**new-security-context** header. `handle_downlink` (`orchestrator.rs`)
special-cases it: an SMC with `sht.is_new_security_context()` is verified with the
keys signalled *inside* the message, so it routes straight to
`handle_security_mode_command` even though the context is not yet active. That
handler enforces, in order (cited against TS 24.501 §5.4.2.3 / TS 33.501 §6.7.2
per code comments):

1. **Anti-bidding-down**: the *replayed* UE security capabilities must equal what
   the UE sent (`ea_cap`/`ia_cap`), else Security Mode Reject with cause **#23**
   (`UeSecurityCapabilitiesMismatch`).
2. **Algorithm validity**: the selected NEA/NIA must decode and be within the
   UE's advertised capabilities; NIA0 (null integrity) is refused outside an
   emergency context.
3. **Partial context present**: a KAMF from authentication is required.
4. **ABBA**: if the SMC carries a different ABBA, KAMF is re-derived
   (`derive_kamf`).
5. **Derive + verify before activating**: `derive_nas_keys` produces
   `Knas_int`/`Knas_enc`, then `verify_nas_mac` checks the SMC's NAS-MAC — and
   only on success is the context `activate`d. A MAC failure rejects with the keys
   **not** activated.

The UE then builds **Security Mode Complete**, adding IMEISV when requested and —
critically — the **replayed initial Registration Request** in the NAS message
container (`complete.nas_message_container = self.last_registration_request`, the
RINMR of TS 24.501 §5.4.2.3). This is what carries the non-cleartext IEs
(requested NSSAI, last-visited TAI) that were withheld from the unprotected
step-4 message. It is protected with header
`IntegrityProtectedAndCipheredWithNewSecurityContext` via `protect_uplink`.
After activation, `handle_downlink` discards any unprotected NAS message
(TS 24.501 §4.4.4.2). Ciphering/integrity primitives are in
`nextgsim-crypto` (`nia.rs`/`nea.rs`); the algorithm negotiation and MAC checks
are exercised by `test_smc_success_activates_and_replays_registration_request`
in `orchestrator.rs`.

### AS security (KgNB → RRC keys)

On the gNB side, `activate_as_security` (`nextgsim-gnb/src/ngap/task.rs`) selects
the AS algorithms from the UE Security Capabilities bitmaps
(`select_as_algorithms`: prefer NEA2/NIA2, and integrity never falls back to
NIA0), derives the four AS keys — `K_RRCenc`, `K_RRCint`, `K_UPenc`, `K_UPint` —
from the KgNB in the Security Key IE using `derive_rrc_up_key`
(`nextgsim-crypto/src/kdf.rs`, cited against TS 33.501 Annex A.8 per code
comments), stores them on the UE context, and sends an **RRC SecurityModeCommand**
on SRB1 (DL-DCCH) via `encode_security_mode_command`.

On the UE side, the KgNB is derived independently by the NAS plane —
`derive_kgnb_for_as_security` (`orchestrator.rs`) computes
`KgNB = derive_kgnb(KAMF, uplink-NAS-COUNT, 0x01)` (TS 33.501 §6.9.4.1 per code
comments), so both ends arrive at identical RRC keys. The UE RRC handler
`handle_as_security_mode_command` (`nextgsim-ue/src/rrc/task.rs`) then
`derive_from_kgnb`s and replies SecurityModeComplete.

> **Caveat (honesty):** the UE-side AS SecurityModeCommand handling is gated
> behind the compile-time flag `I5_UE_AS_SECURITY` (`nextgsim-ue/src/rrc/security.rs`),
> **default off**. With the flag off — the matched-sim default and the E2E path —
> the gNB still derives AS keys and emits the RRC SecurityModeCommand, but the
> UE's legacy DL-DCCH nibble dispatcher misroutes the gNB's ASN.1 SMC rather than
> completing typed AS security (this exact behavior is pinned as the red/green
> baseline in `tests/src/rrc_handshake.rs`). Because RLS/RLC do not actually
> cipher SRBs/DRBs in the sim, registration and the user plane still complete;
> full UE↔gNB AS-security completion is a tracked Wave-6 residue (I5), not a
> claim of this default path.

## Simplifications & known gaps

Grounded in the code, these are the honest deviations from a textbook TS 23.502
registration:

- **Simulated radio and system information.** RLS models signal strength as
  `-distance` (`GnbCellTracker::estimate_dbm`) over UDP; there is no MIB/SIB
  broadcast. The UE *fabricates* SIB1/MIB locally in
  `provide_simulated_system_info` (`nextgsim-ue/src/rrc/task.rs`) so the cell is
  selectable — TS 38.304 cell selection runs, but on synthesised inputs.
- **5G-AKA only on the live path.** `handle_authentication_request` runs
  MILENAGE-based 5G-AKA. `handle_eap_aka_prime_challenge` is only reached if the
  Authentication Request omits RAND/AUTN and carries an EAP-message IE, which the
  matched core does not do.
- **AS security default-off on the UE (I5).** See the caveat above:
  `I5_UE_AS_SECURITY` gates the UE's typed AS SMC handling; the default path
  leaves AS security key-derived-but-not-completed on the UE, tolerated because
  the simulated air interface does not cipher.
- **RRC transaction ids pinned to 0.** Both the gNB's RRCSetup/SMC tids and the
  UE's echoed RRCSetupComplete tid are pinned to `0` while
  `C5_TYPED_DCCH_DISPATCH` (`nextgsim-gnb/src/rrc/transaction.rs`) is off, to stay
  nibble-safe on the legacy dispatchers. The full per-UE `0..3` cycle only lands
  with C5.
- **Bespoke DCCH framing alongside ASN.1.** The gNB accepts both real UPER
  UL-DCCH and a bespoke fallback (`[0x04, tid, 0x01, NAS…]`), and emits DL NAS as
  a bespoke `[0x04,0x00,0x00,NAS…]` container unless C5 is on. These legacy paths
  are slated for retirement (C6) and exist for matched-sim interop.
- **Identity procedure normally skipped.** The matched UE sends its SUCI directly
  in the initial Registration Request, so no Identity Request/Response runs on the
  happy path.
- **Cooperative peer.** All of the above is exercised against nextgcore, not an
  arbitrary commercial core/RAN. The 6G/AI signalling routed by the RRC task
  (`SixgAiMlInference`, `SixgIsacSensingData`, …) is a non-normative research
  prototype and plays no part in registration.

## How it is validated

**Unit tests (per-crate `cargo test`).** The wire builders and decision logic are
tested independently of any network:

- `nextgsim-rls/src/cell_search.rs` — `test_ue_cell_search_discovery`,
  `test_gnb_tracker_detection`, `test_gnb_tracker_weak_signal`, `test_best_cell`
  (heartbeat discovery, dBm gating, best-cell selection).
- `nextgsim-ue/src/rrc/cell_selection.rs` — `test_cell_selection_suitable_cell`,
  `test_cell_selection_acceptable_cell`, `test_cell_selection_barred_cell_rejected`,
  plus the TS 23.122 PLMN-selection suite (`test_plmn_selection_*`).
- `nextgsim-ue/src/nas/mm/orchestrator.rs` —
  `test_registration_request_arms_t3510_and_includes_capabilities`,
  `test_smc_success_activates_and_replays_registration_request`,
  `test_registration_accept_with_guti_sends_registration_complete`,
  `test_plain_registration_accept_discarded` (the §4.4.4.2 unprotected-message
  gate), and the reject-cause table tests.
- `nextgsim-ue/src/nas/mm/suci.rs` — `test_null_scheme_exact_bytes`,
  `test_profile_a_conceals_and_deconceals`, `test_profile_b_conceals_and_deconceals`
  (golden SUCI bytes vs TS 33.501 Annex C vectors).
- `nextgsim-gnb/src/rrc/connection.rs` —
  `test_rrc_setup_request_emits_golden_srb1_pdu_tid0`,
  `test_rrc_setup_complete_tid_mismatch_is_discarded`.
- `nextgsim-gnb/src/rrc/task.rs` — `test_ul_dcch_asn1_rrc_setup_complete_accepted`,
  `test_ul_dcch_bespoke_rrc_setup_complete_accepted`, and the typed-dispatch tests.
- `nextgsim-ngap/src/capture_tests.rs` — `test_initial_ue_message_encoding_structure`,
  `test_decode_initial_ue_message_capture`, `test_initial_ue_message_rrc_causes`,
  `test_initial_ue_message_large_nas_pdu` (APER encode/decode of InitialUEMessage).

**In-process strict-peer integration.** `tests/src/rrc_handshake.rs` drives the
**real** gNB and UE RRC task handlers in one process (no mocks): UE
RRCSetupRequest → gNB RRCSetup → UE RRCSetupComplete → the resulting
`NgapMessage::InitialNasDelivery` whose NAS bytes are asserted byte-for-byte equal
to the original Registration Request. `tests/src/ue_registration.rs` and
`tests/src/e2e_scenario.rs` (`test_e2e_ue_registration_flow`) exercise the
registration path through the task graph.

**Matched-simulator docker E2E.** `nextgsim/docker/docker-compose.yml` brings up
the prebuilt `nr-gnb`/`nr-ue` (`docker/binaries/`) on the `nextgcore_core`
network and runs them against nextgcore; the end-to-end driver script lives in
the nextgcore repo (`nextgcore/docker/rust/e2e.sh`). That suite asserts the
registration flow through source-anchored log lines (`Sending Registration
Request` … `Registration Accept: UE is now …` … `Sending Registration Complete`)
and then verifies the data plane by pinging **through the UE's GTP-U tunnel**,
finishing at an active PDU session — **84/84 green** as of the last baseline. This
is functional validation against a cooperative peer; it is not third-party
conformance certification. See [Getting Started](../getting-started.md) for the
run recipe and [Configuration Reference](../configuration.md) for the knobs that
must match between gNB and UE for registration to succeed.
