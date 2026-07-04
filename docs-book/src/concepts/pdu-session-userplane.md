# PDU Sessions and the User-Plane Data Path

This chapter traces how the nextgsim UE and gNB establish a PDU session and
then carry user-plane traffic, **as the Rust code actually runs it** — not as
TS 23.502 draws the reference flow. It follows one PDU session from the UE's
5GSM *PDU Session Establishment Request*, through the gNB's NGAP
`PDUSessionResourceSetup` handling, the GTP-U N3 tunnel toward the UPF, and the
RRC/DRB signalling, and then walks a single IP packet in each direction across
the simulated radio link (RLS) and the UE's TUN device. Every step names the
crate, file, and function or struct that performs it, so you can open the source
and read along. The peer 5G core is [nextgcore](../reference.md); nextgsim is the
matched simulator that drives it over N2 (NGAP/SCTP) and N3 (GTP-U).

> **Honesty note:** This flow is validated by nextgsim's own unit tests and by a
> **matched-simulator** docker E2E — the project drives its *own* gNB and UE
> against nextgcore, and the UE→UPF GTP-U data plane has been verified with 0%
> packet loss (project record: 84/84 green, 2026-07-02). It is **not**
> third-party conformance-certified, and it matches a cooperative peer rather
> than an arbitrary commercial UPF/UE. Several deliberate simplifications exist
> (the UE acknowledges the DRB `RRCReconfiguration` without installing a
> per-DRB user-plane path; the simulated radio is UDP + an RLC-UM segmentation
> layer keyed by PDU-session ID; the `--no-routing-config` flag is currently
> inert). 6G GTP-U extension headers (TSN, in-network compute) are non-normative
> research prototypes with no frozen Rel-20 Stage-3 to conform to. 3GPP TS
> numbers below are quoted only where they appear in this repo's source
> comments, phrased "per code comments".

## Components involved

| Concern | Crate | Key file(s) | Role |
|---|---|---|---|
| UE 5GSM state machine | `nextgsim-ue` | `src/nas/sm/orchestrator.rs`, `src/nas/sm/procedure.rs` | Builds/parses PDU session messages; PSI/PTI allocation; T3580/T3581/T3582 |
| UE TUN device | `nextgsim-ue` | `src/tun/interface.rs`, `src/tun/task.rs`, `src/tun/config.rs` | Creates `uesimtun<N>`, reads uplink / writes downlink IP packets |
| UE↔gNB radio link | `nextgsim-rls` | `src/protocol.rs`, `src/transport.rs` | RLS-over-UDP frames for RRC and user data |
| RLC segmentation | `nextgsim-rlc` | `src/entity.rs` | Per-session RLC-UM (SN12) segment/reassemble |
| gNB NGAP / N2 | `nextgsim-gnb` | `src/ngap/task.rs` | `PDUSessionResourceSetup` + `InitialContextSetup`, DRB reconfig |
| NGAP transfer codecs | `nextgsim-ngap` | `src/procedures/transfer.rs` | APER encode/decode of the N2 SM transfer containers |
| gNB GTP-U / N3 | `nextgsim-gnb`, `nextgsim-gtp` | `src/gtp/task.rs`; `src/tunnel.rs`, `src/codec.rs` | GTP-U encap/decap, TEID/tunnel management |
| RRC / DRB messages | `nextgsim-rrc` | `src/procedures/rrc_reconfiguration.rs` | Builds the `RadioBearerConfig` (SDAP + DRB) for the DRB |

Related reading: the [UE Registration Flow](registration-flow.md) that must
complete before a session can be established; the
[gNB architecture](../architecture/gnb.md) and [UE architecture](../architecture/ue.md)
task diagrams; the [Configuration reference](../configuration.md) for the
`sessions:` and TUN knobs; and the [protocol gap analysis](../gaps/gap-protocols.md)
and [AI/6G stack](ai-6g-stack.md) for the non-normative extension surfaces.

## Control-plane: establishing the session

The UE must already be RM-REGISTERED (see [Registration Flow](registration-flow.md))
and hold an active NAS security context, because the 5GSM message travels inside
an integrity-protected UL NAS Transport. A session is started either
automatically for each entry in the UE config's `sessions:` list, or on demand
via the `ps-establish` CLI command (see [nr-cli Reference](../cli-reference.md)).

```
UE(nas/sm)      UE(rrc/rls)      gNB(rls/rrc)      gNB(ngap)      AMF/SMF        gNB(gtp)/UPF
   | PDU Sess Est Req |               |                |             |               |
   |----------------->| UL-DCCH (RLS) |                |             |               |
   |                  |-------------->| UplinkNAS      |             |               |
   |                  |               |--------------->| UL NAS Tx   |               |
   |                  |               |                |------------>| SMF selects   |
   |                  |               |                |             | UPF, N3 F-TEID|
   |                  |               |   ICS / PDUSessResourceSetup |               |
   |                  |               |<---------------|<------------|               |
   |                  |               |  setup_one_pdu_session()     | GTP SessCreate|
   |                  |               |                |------------------------------>|
   |                  |  RRCReconfig (DRB) + NAS(Accept)|             |               |
   |                  |<--------------|<---------------|             |               |
   | PDU Sess Est Acc |               | Setup Response |             |               |
   |<-----------------|               |--------------->|------------>|               |
   | TUN uesimtunN up |               |                |             |               |
```

1. **UE builds the PDU Session Establishment Request.**
   `SmOrchestrator::start_establishment` (`nextgsim-ue/src/nas/sm/orchestrator.rs`)
   allocates the lowest free PDU session identity 1–15 (`allocate_psi`) and a
   procedure transaction identity 1–254
   (`ProcedureTransactionManager::allocate` in `src/nas/sm/procedure.rs`). It
   constructs a `PduSessionEstablishmentRequest` (from `nextgsim-nas`) with the
   requested PDU session type, SSC mode (`SscModeValue::SscMode1`), and a 5GSM
   capability octet of `0x00` (no reflective QoS / no multi-homed IPv6, per
   inline comment). The parameters come from the UE config via
   `SmSessionParams::from_config` (default: one IPv4 `internet` session on the
   first configured slice). Session state moves to `PsState::ActivePending` and
   T3580 is armed (`SM_TIMER_T3580`). Logs `Sending PDU Session Establishment
   Request: PSI=…, PTI=…, T3580 started`.

2. **UE wraps it in UL NAS Transport.** `build_ul_transport` (same file) builds a
   plain `UlNasTransport` with `PayloadContainerType::N1SmInformation`, the PDU
   session ID, the *request type* (`InitialRequest`, or
   `InitialEmergencyRequest` for an emergency session), the S-NSSAI, and the DNN
   (encoded by `encode_dnn_value` → `IeDnn::from_string`, per code comments TS
   24.501 §9.11.2.1B). The orchestrator is sans-IO: it returns
   `SmOutput::SendNasPdu(bytes)` for the caller to protect and transmit.

3. **UE protects and transmits the NAS PDU.** `process_sm_outputs`
   (`nextgsim-ue/src/main.rs`) takes the `SmOutput::SendNasPdu`, applies the MM
   NAS security context via `orch.protect_if_active(plain)` (per code comments
   TS 24.501 §4.4 / §5.4.5), and hands it to the RRC task as
   `RrcMessage::UplinkNasDelivery`. RRC sends it on SRB (UL-DCCH) over RLS to the
   serving gNB.

4. **gNB relays the uplink NAS to the AMF.** The gNB RRC task forwards the
   received NAS to the NGAP task with `NgapMessage::UplinkNasDelivery`
   (`nextgsim-gnb/src/rrc/task.rs`, `send_uplink_nas_delivery`), which the NGAP
   task encodes as an NGAP `UplinkNASTransport` toward the AMF. The AMF/SMF
   select a UPF and its N3 F-TEID (this happens inside nextgcore).

5. **gNB receives the N2 setup instruction.** The NGAP dispatcher in
   `nextgsim-gnb/src/ngap/task.rs` (around line 1923) tries each procedure
   decoder in turn. The SMF's decision arrives one of two ways, and both funnel
   into the same per-session routine:
   - inside an **`InitialContextSetupRequest`** carrying a
     `PDUSessionResourceSetupListCxtReq` — handled by
     `handle_initial_context_setup_request`. Per the `amfg-01` code comment,
     "Real AMFs (e.g. Open5GS) deliver the first PDU session this way during
     registration+PDU", so the N3 tunnel is set up here, not only on the
     dedicated procedure.
   - as a dedicated **`PDUSessionResourceSetupRequest`** — handled by
     `handle_pdu_session_resource_setup`.

6. **gNB sets up one PDU session.** `setup_one_pdu_session`
   (`nextgsim-gnb/src/ngap/task.rs`) does the real work per session:
   - **Decode the transfer container.** `decode_setup_request_transfer`
     (`nextgsim-ngap/src/procedures/transfer.rs`) APER-decodes the
     `PDUSessionResourceSetupRequestTransfer` (per code comments TS 38.413
     §9.3.4.1), yielding the UPF UL F-TEID (`request.ul_tunnel.teid` /
     `.address`), the PDU session type, and the requested QoS flows. The first
     flow's QFI is taken as the session's default flow.
   - **Allocate the gNB downlink TEID.** `next_downlink_teid` (same file) is a
     simple monotonically incrementing `u32` counter starting at 1.
   - **Store the session** in the UE's NGAP context (`ctx.add_pdu_session`).
   - **Create the GTP-U tunnel.** It sends `GtpMessage::SessionCreate` with a
     `PduSessionResource { uplink_teid = upf_teid, downlink_teid = gnb_teid,
     upf_address }` to the GTP task (see the next section).
   - **Forward the piggybacked NAS PDU** (the *PDU Session Establishment
     Accept*, if present) to the RRC task as `RrcMessage::NasDelivery`.
   - **Establish the DRB** via `establish_drb` (step 7).
   - **Build the response transfer.** `encode_setup_response_transfer`
     (`transfer.rs`) APER-encodes a `PDUSessionResourceSetupResponseTransfer`
     (per code comments TS 38.413 §9.3.4.2) carrying the gNB DL F-TEID and the
     accepted QFIs; this is returned in a `PDUSessionResourceSetupResponse` (or
     `InitialContextSetupResponse`) to the AMF.

7. **gNB signals the DRB with an RRCReconfiguration.** `establish_drb`
   (`nextgsim-gnb/src/ngap/task.rs`) allocates one DRB per PDU session (`drb_id =
   psi.clamp(1, 32)`, DTCH `lcid = 3 + drb_id`) and calls
   `build_drb_reconfiguration_params`
   (`nextgsim-rrc/src/procedures/rrc_reconfiguration.rs`). That builds a real
   ASN.1/UPER `RadioBearerConfig` — a `DRB-ToAddMod` with an `SDAP-Config` keyed
   to the PDU session id and the accepted QFIs — plus a matching
   `CellGroupConfig` (one RLC bearer), and encodes them with `encode_rrc`. Per
   the `Wave-6 C4-final` code comment the `rrc_transaction_id` is **pinned to 0
   on the wire** while the typed DL-DCCH dispatcher is off (a non-zero tid would
   shuffle the leading-byte nibble the UE's legacy dispatcher routes on). The
   PDU is handed to the RRC task and sent to the UE on SRB. Logs `Sent
   RRCReconfiguration establishing DRB … for PDU session …`.

8. **UE accepts the session.** The downlink NAS reaches the UE's SM orchestrator
   via `SmOrchestrator::handle_dl_nas_transport` →
   `handle_establishment_accept` (`nextgsim-ue/src/nas/sm/orchestrator.rs`). It
   validates the PTI/PSI against the pending transaction
   (`validate_response_pt`), strictly decodes `PduSessionEstablishmentAccept`
   (only the spec-conformant wire format of Table 8.3.2.1.1 is accepted, per
   code comment), extracts the IPv4 address from the PDU address IE, stops
   T3580, moves the session to `PsState::Active`, and stores the selected type,
   SSC mode, authorized QoS rules, and session-AMBR. It returns
   `SmOutput::SessionEstablished { psi, ipv4 }`. Logs `PDU Session Establishment
   Accept: PDU session … established: type …, IPv4 …`.

9. **UE acknowledges the DRB reconfiguration.** The `RRCReconfiguration` reaches
   `handle_rrc_reconfiguration` (`nextgsim-ue/src/rrc/task.rs`). For a
   non-handover reconfiguration the UE extracts the transaction id (a simplified
   `bytes[1]`) and replies with an `RRCReconfigurationComplete`
   (`build_reconfiguration_complete`) — **it does not decode the
   `RadioBearerConfig` or install a per-DRB user-plane bearer.** See
   *Simplifications & known gaps*.

10. **UE brings up the TUN device.** Back in `process_sm_outputs`
    (`nextgsim-ue/src/main.rs`), `SmOutput::SessionEstablished` with an IPv4
    address sends `TunMessage::CreateInterface { psi, address, netmask
    255.255.255.0 }` to the TUN task. `TunTask::create_interface`
    (`nextgsim-ue/src/tun/task.rs`) calls
    `TunInterface::create_and_configure` (`src/tun/interface.rs`), which uses the
    `tun-rs` crate to create an L3 device named `uesimtun<N>` (default prefix
    `uesimtun`, MTU 1400 from `TunTaskConfig::default`), assigns the address and
    netmask, brings it up, and spawns a reader task. Logs `TUN interface created
    … for PSI …`. The session is now usable end to end.

## User-plane: the data path

Once the TUN device is up, an application binds to the UE's assigned IP and
sends IP packets. Two distinct transports carry a packet across the simulator:
the **RLS** link (UDP, UE↔gNB, standing in for the radio) and the **GTP-U** N3
tunnel (gNB↔UPF). They are stitched together inside the gNB.

```
        UPLINK  (app -> internet)                DOWNLINK (internet -> app)
  app                                       app
   |  IP packet                              ^  IP packet
   v                                         |
 uesimtunN  (tun/interface.rs recv)        uesimtunN (tun/task.rs write_data)
   |  TunAppMessage::UplinkData              ^  TunMessage::WriteData
   v                                         |
 UE main.rs  --RlsMessage::DataPduDelivery-> UE NAS/main.rs  <-NasMessage::UplinkDataDelivery--
   |                                         ^
   v  UE rls/task.rs handle_data_pdu_delivery|  UE rls/task.rs TransportEvent::DataReceived
   |    RLC-UM segment (SN12)                |    RLC-UM reassemble
   v    RLS PduTransmission(Data) / UDP      |    RLS PduTransmission(Data) / UDP
 ========================= simulated radio (RLS over UDP) =========================
   |                                         ^
   v  gNB rls/task.rs handle_uplink_data     |  gNB rls/task.rs (DownlinkData)
   |    RLC reassemble -> full SDU           |    RLC segment
   v  GtpMessage::DataPduDelivery            |  RlsMessage::DownlinkData
   |                                         ^
   v  gNB gtp/task.rs handle_uplink_data     |  gNB gtp/task.rs handle_downlink_gpdu
   |    encapsulate_uplink -> GTP-U G-PDU    |    decapsulate_downlink (by DL TEID)
   v    UDP :2152 to UPF                     |    UDP :2152 from UPF
 ============================= N3 (GTP-U to/from UPF) =============================
                              UPF (nextgcore)
```

**Uplink, hop by hop:**

1. **TUN read.** The reader task spawned in `TunTask::create_interface`
   (`nextgsim-ue/src/tun/task.rs`) calls `TunReader::recv`
   (`src/tun/interface.rs`), validates the packet with `IpPacket::parse`, and
   emits `TunAppMessage::UplinkData { psi, data }`.
2. **UE app fan-out.** The handler in `nextgsim-ue/src/main.rs` (the
   `TunAppMessage::UplinkData` arm) sends `NasMessage::InitiateServiceRequest`
   (so an IDLE UE transitions to CONNECTED before sending) and forwards the
   packet to RLS as `RlsMessage::DataPduDelivery { psi, pdu }`.
3. **UE RLS + RLC.** `handle_data_pdu_delivery`
   (`nextgsim-ue/src/rls/task.rs`) submits the IP packet as an SDU to a per-PSI
   RLC-UM entity (`rlc_entity_for`), pulls out RLC PDUs with `build_pdu(1500)`,
   and wraps each in an RLS `PduTransmission` via
   `RlsTransport::create_data_transmission(psi, …)`
   (`nextgsim-rls/src/transport.rs`). That helper sets the frame's `payload`
   field to the **PSI** and `pdu_id = 0` (user data is unacknowledged). The
   frame goes over UDP to the serving cell.
4. **gNB RLS + RLC.** The gNB RLS task routes `PduType::Data` frames to
   `handle_uplink_data` (`nextgsim-gnb/src/rls/task.rs`), which reads
   `psi = pdu.payload`, feeds the bytes into the per-UE RLC entity, and on each
   fully reassembled SDU sends `GtpMessage::DataPduDelivery { ue_id, psi, pdu }`
   to the GTP task.
5. **gNB GTP-U encap.** `handle_uplink_data` (`nextgsim-gnb/src/gtp/task.rs`)
   drops non-IPv4 packets, then (via `TunnelManager::encapsulate_uplink` in
   `nextgsim-gtp/src/tunnel.rs`) builds a GTP-U G-PDU carrying a UL PDU Session
   Information container (QFI; default QFI 1 when none is assigned, per code
   comments TS 38.415 §5.5.2.2 / TS 29.281 §5.2.2.7) and `send_to`s it over UDP
   :2152 to the UPF's N3 endpoint.

**Downlink** reverses the path: the UPF's G-PDU arrives on the gNB GTP-U socket
(`handle_udp_receive` → `handle_downlink_gpdu`, `src/gtp/task.rs`),
`TunnelManager::decapsulate_downlink` (`nextgsim-gtp/src/tunnel.rs`) finds the
session by **downlink TEID** and lifts out the payload (plus any QFI/RQI from the
DL PDU Session container). The gNB emits `RlsMessage::DownlinkData`, the gNB RLS
task RLC-segments it back to the UE, the UE RLS task reassembles it
(`TransportEvent::DataReceived`) and passes it up as
`NasMessage::UplinkDataDelivery` (the name is direction-agnostic — it means
"deliver up the stack"), and the UE NAS handler in `main.rs` writes it to the
TUN with `TunMessage::WriteData` → `TunTask::write_data` →
`TunWriter::send`.

## GTP-U and TEID handling

The GTP-U implementation lives in the `nextgsim-gtp` crate: `src/codec.rs`
(header + extension-header wire format) and `src/tunnel.rs` (tunnel/session
bookkeeping). The gNB's GTP task (`nextgsim-gnb/src/gtp/task.rs`) owns the UDP
socket and the `TunnelManager`.

| Aspect | Behaviour | Reference |
|---|---|---|
| Transport | Single UDP socket bound to `gtp_ip:2152` (`GTP_U_PORT = 2152`) | `gtp/task.rs` `init_udp_socket`; `tunnel.rs` |
| G-PDU message type | `255` (`GtpMessageType::GPdu`) | `codec.rs` `GtpMessageType` |
| Session key | `(ue_id << 32) | psi` in a `HashMap` | `tunnel.rs` `make_session_key` |
| Uplink target | UPF F-TEID + address from the NGAP transfer container | `ngap/task.rs` `setup_one_pdu_session`; `gtp/task.rs` `handle_session_create` |
| gNB DL TEID | Monotonic `u32` counter from 1 | `ngap/task.rs` `next_downlink_teid` |
| Downlink lookup | By DL TEID (`downlink_teid_map`) | `tunnel.rs` `find_by_downlink_teid` |
| QoS metadata | PDU Session Container ext header `0x85`; QFI mandatory, RQI on DL | `codec.rs` `PduSessionInfo`; `tunnel.rs` `encapsulate_uplink`/`decapsulate_downlink` |
| Echo | Answers GTP-U Echo Request with Echo Response (+recovery IE) | `gtp/task.rs` `handle_echo_request` |

**TEID directions can be confusing**, so note the mapping the code uses
(`gtp/task.rs` `handle_session_create`): a session's *uplink tunnel* is
`(UPF's TEID → UPF address)` — where the gNB sends uplink G-PDUs — and its
*downlink tunnel* is `(gNB's own TEID → gNB address)` — where the UPF sends
downlink. The gNB advertises its own TEID/address in the setup response transfer
(using `gtp_advertise_ip` if configured, else `gtp_ip`).

**Auto-create fallback and loopback.** If a data PDU arrives for a session that
was never signalled, `handle_uplink_data` (`gtp/task.rs`) creates one on the
fly: `auto_create_upf_session` derives a TEID by bit-packing `ue_id`/`psi` and
forwards to the config `upf_addr`, while `auto_create_loopback_session` is used
when `upf_addr` is unset. `loopback_mode` is enabled whenever the gNB config has
no `upf_addr` (`GtpTask::new`); in that mode `handle_loopback_data` echoes ICMP
Echo Requests straight back to the UE (swapping src/dst and recomputing the IP
and ICMP checksums), letting the data plane be exercised without any core. The
`nextgsim-gtp` codec also defines forward-looking **6G** extension headers —
`TsnMarker` (`0xE1`) and `InNetworkCompute` (`0xE2`) — which are non-normative
research prototypes (see [AI/6G stack](ai-6g-stack.md)).

## Simplifications & known gaps

Grounded in the code, these are the honest deviations from a textbook TS
23.501/23.502 user plane:

- **The UE does not install the DRB it is told to.** `handle_rrc_reconfiguration`
  (`nextgsim-ue/src/rrc/task.rs`) acknowledges a non-handover
  `RRCReconfiguration` with `RRCReconfigurationComplete` but never decodes the
  `RadioBearerConfig`/`SDAP-Config` or maps QFIs to a bearer. The gNB *emits* a
  real ASN.1/UPER DRB config (`build_drb_reconfiguration_params`,
  `nextgsim-rrc`), but user-plane data is switched purely by **PDU-session ID**,
  not by DRB/LCID/QFI.
- **The "radio" is UDP plus RLC-UM.** RLS frames (`nextgsim-rls`) ride UDP with
  no MAC/PHY, HARQ, or PDCP ciphering on the user plane. Segmentation/reassembly
  is a per-session RLC-UM (SN12) entity (`nextgsim-rlc`); the PSI is carried in
  the RLS frame's `payload` field (`create_data_transmission`), not in a real
  logical-channel identity.
- **DL QoS metadata is surfaced but not acted on.** `decapsulate_downlink`
  (`nextgsim-gtp/src/tunnel.rs`) extracts the DL QFI/RQI, but per the `amfg-09`
  code comment they drive nothing toward the UE yet ("Until the SDAP/DRB layer
  is wired … the session is selected by PSI and the QoS metadata is surfaced for
  diagnostics / reflective QoS").
- **The `--no-routing-config` (`-r`) flag is currently inert.** The CLI option
  and the `TunConfig::configure_routing` field exist
  (`nextgsim-ue/src/main.rs`, `src/tun/config.rs`), but the option is only logged
  (`TUN routing configuration: disabled`) and never threaded into TUN setup — no
  `ip route` is ever programmed. It is a UERANSIM-parity stub. In practice you
  route to a UE by binding to its `uesimtun<N>` address/interface.
- **TEID allocation is not a real F-TEID pool.** The gNB DL TEID is a plain
  incrementing counter (`next_downlink_teid`), and the auto-create fallback
  bit-packs `ue_id`/`psi` into the TEID (`gtp/task.rs`).
- **On the NGAP transfer containers, no shortcut is taken.** The N2 SM transfer
  containers *are* encoded/decoded with real X.691 Aligned-PER
  (`nextgsim-ngap/src/procedures/transfer.rs` — "Real APER encode/decode …").
  The one cooperative-peer nuance, documented in that module (`ngap-04`): the
  outer extensible-SEQUENCE extension-presence bit is matched bit-for-bit to the
  nextgcore `nextgcore-ngap` peer's emission. The UE likewise accepts **only** the
  strict, spec-conformant *Establishment Accept* wire format (Table 8.3.2.1.1);
  a legacy/non-conformant emission is rejected with 5GSM Status #96
  (`handle_establishment_accept`, `nextgsim-ue/src/nas/sm/orchestrator.rs`).
- **Cooperative peer.** Everything above is exercised against the matched
  nextgcore (or, on the exit-gate, a real Open5GS), not an arbitrary commercial
  UPF; SMF-side session policy/QoS derivation happens in nextgcore, not here.

## How it is validated

**Unit tests (per-crate `cargo test`).** The wire builders and state machines
are tested independently of any network:

- `nextgsim-ue/src/nas/sm/orchestrator.rs` — e.g.
  `test_establishment_request_wrapped_in_ul_nas_transport`,
  `test_psi_and_pti_are_allocated_not_hardcoded`,
  `test_establishment_accept_activates_session`,
  `test_legacy_core_accept_now_rejected`, the per-cause reject/back-off tests,
  and the T3580 retransmission/exhaustion tests.
- `nextgsim-gtp/src/tunnel.rs` — `test_encapsulate_uplink`,
  `test_encapsulate_uplink_qfi_container_roundtrip`,
  `test_decapsulate_downlink_with_qfi_container`, `test_decapsulate_unknown_teid`
  (TEID lookup, PDU Session Container round-trip).
- `nextgsim-gnb/src/gtp/task.rs` — `test_handle_session_create_with_context`,
  `test_handle_session_release`, `test_handle_ue_context_release`.

**Integration tests** (the `integration-tests` crate, `nextgsim/tests/src/`)
drive the tasks against a `MockAmf`: `pdu_session.rs`
(`test_pdu_session_establishment`), `user_plane.rs` (GTP-U encap/decap flow),
`e2e_scenario.rs` (full gNB→UE→mock-core scenario), `multi_ue.rs`, and
`ue_registration.rs`.

**Matched-simulator docker E2E.** `nextgsim/docker/docker-compose.yml` brings up
the nextgsim gNB and UE on the same Docker bridge as nextgcore and runs the full
registration + PDU-session + data-plane path; the single-command entrypoint that
chains build and assertions lives on the core side
(`nextgcore/docker/rust/e2e.sh`). The suite pings **through the UE's GTP-U
tunnel** and finishes at `UE PDU session 1 ACTIVE with IPv4 address`; the project
record has this at **84/84 green (2026-07-02) with UE→UPF GTP-U at 0% packet
loss**. A stricter **exit-gate** (`nextgsim/docker/exit-gate/`) runs the gNB
(kernel SCTP) + UE against a real Open5GS 2.7 SA core as an independent peer.

All of this is functional validation against a cooperative or reference peer —
it is **not** third-party conformance certification.
