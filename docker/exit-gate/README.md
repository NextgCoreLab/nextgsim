# Strict-Peer Exit Gate (T6.3) — direction (a): our gNB+UE → a real Open5GS 2.7 5G SA core

Self-contained Docker setup that runs our sim gNB (kernel SCTP, `--features
kernel-sctp`) and sim UE against a **real, full Open5GS 2.7 5G SA core**
(nrf/scp/amf/ausf/udm/udr/pcf/bsf/nssf/smf/upf + mongo) — the independent peer
that proves the matched-simulator crutch is gone. PLMN 999-70, DNN `internet`,
S-NSSAI sst:1 (Open5GS defaults, matched by our gNB/UE).

The Open5GS NFs use the openverso image's default configs; only udr/pcf (db_uri)
and smf (UPF IP + Diameter off) are patched (`open5gs/*.yaml`). Every service is
pinned to a static IP (Docker IPAM otherwise hands a dynamic NF the amf/upf
addresses). The subscriber (IMSI 999700000000001, Open5GS test K/OPc) is seeded
into mongo via `mongosh` (no `open5gs-dbctl` in the image).

## Run
```bash
# 1. Build the Linux binaries (gitignored under binaries/):
cd ../..   # nextgsim root
CACHE=~/.cache/nextgsim-docker-build; mkdir -p "$CACHE"/{target,registry,git}
docker run --rm -v "$PWD":/work -w /work -v "$CACHE/target":/target \
  -v "$CACHE/registry":/usr/local/cargo/registry -v "$CACHE/git":/usr/local/cargo/git \
  -e CARGO_TARGET_DIR=/target rust:latest bash -c \
  'apt-get update -qq && apt-get install -y -qq libsctp-dev && \
   cargo build -p nextgsim-gnb --features kernel-sctp && cp /target/debug/nr-gnb binaries/nr-gnb-kernel && \
   cargo build -p nextgsim-ue && cp /target/debug/nr-ue binaries/nr-ue-linux'
# 2. Bring up the full core + gNB:
cd docker/exit-gate
docker compose up -d --build nrf scp amf ausf udm udr pcf bsf nssf upf smf gnb
# 3. Provision the subscriber (IMSI 999700000000001, K/OPc test keys):
docker exec eg-mongodb mongosh open5gs --quiet --eval '<see git history / runbook>'
# 4. Bring up the UE:
docker compose up -d --build ue
docker logs eg-ue -f       # UE: registration / PDU / data plane
docker logs eg-amf -f      # Open5GS AMF
docker compose down        # teardown
```

## Status — 2026-06-15

**MILESTONE 1 (SCTP/association) — PASSED.** Confirmed against a real Open5GS AMF:
- Our **kernel SCTP backend (T0.2) works at runtime**: `Kernel SCTP association
  established: 172.30.0.100:50850 -> 172.30.0.5:38412`.
- **PPID 60 is correct on the wire**: `sent 60 bytes ... PPID 60 (wire be: 0x3c000000)`
  (network byte order = Open5GS `htobe32`).
- **Open5GS accepted us**: `gNB-N2 accepted[172.30.0.100]` · `[Added] Number of
  gNBs is now 1` · `max_num_of_ostreams: 2`. The transport crutch is gone.

**MILESTONE 2 (NG Setup procedure) — PASSED.** Validated end-to-end against a
real Open5GS 2.7 AMF over kernel SCTP. Capture the N2 link with:
```bash
docker run -d --name eg-cap --net container:eg-amf -v "$PWD/captures":/cap \
  nicolaka/netshoot tshark -i any -f sctp -w /cap/n2.pcap   # then bring up gNB
docker run --rm -v "$PWD/captures":/cap nicolaka/netshoot \
  tshark -r /cap/n2.pcap -d 'sctp.ppi==60,ngap' -V          # dissect
```
The capture (written to `captures/`, gitignored — regenerate by re-running the
gate) shows the full exchange:
- gNB → AMF **NGSetupRequest** (PPID 60) — all 4 IEs decode under Open5GS's
  strict APER decoder (GlobalRANNodeID gNB-ID=1 PLMN 999/70, RANNodeName,
  SupportedTAList TAC=1 SST=01, DefaultPagingDRX). **Our NGAP encoder is
  wire-conformant to an independent peer.**
- AMF → gNB **NGSetupResponse** (PPID 60, successfulOutcome, AMFName
  `open5gs-amf0`) → the gNB marks the AMF context **Connected**.
- Steady bidirectional SCTP HEARTBEAT/HEARTBEAT_ACK — stable association.

### Two real gNB-side bugs this gate found + fixed
The earlier "Open5GS doesn't process our NG Setup" conclusion was an artifact of
reading info-level AMF logs *without a wire capture* — Open5GS **did** respond.
The capture redirected the hunt to the gNB receive path:
1. **Kernel-SCTP receive path was never driven**
   (`nextgsim-gnb/src/sctp/amf_connection.rs`). The SCTP task polls every backend
   through a 10 ms `poll_connections → try_recv` loop, but the kernel backend's
   `try_recv` returned `Ok(None)` unconditionally (`poll` a no-op), so the socket
   was never read — the NGSetupResponse sat unread (kernel SACKed it; the app
   never consumed it). Fixed by adding a real non-blocking
   `KernelSctpAssociation::try_recv` (`nextgsim-sctp/src/kernel.rs`) and wiring
   the kernel arm to it.
2. **Received PPID read as 0** (`nextgsim-sctp/src/kernel.rs`). We enabled the
   modern `SCTP_RECVRCVINFO`, but lksctp's `sctp_recvmsg()` fills `sinfo` only
   from the legacy `SCTP_SNDRCV` cmsg, delivered when `sctp_data_io_event` is set
   via `SCTP_EVENTS`. Without it `sinfo_ppid` stayed 0 (NG Setup still worked — we
   route by content — but the PPID was unvalidated). Fixed by subscribing
   `SCTP_EVENTS` with data_io + association + shutdown events, mirroring Open5GS's
   `ogs_sctp_socket`.

Post-fix the gNB logs `Kernel SCTP received 54 bytes on stream 0 with PPID 60`
and `Received NG Setup Response from AMF 0: name=open5gs-amf0`, no warnings.
**T0.2 kernel-SCTP is now runtime-proven on both send AND receive.**

**MILESTONE 3 (Registration + 5G-AKA + NAS security) — PASSED.** The sim UE
registers against the full real Open5GS core:
`RegistrationAccept (protected=true)` → `5GMM-REGISTERED.NORMAL-SERVICE` →
5G-GUTI assigned → `Registration Complete`. This exercises **5G-AKA (Milenage
with the provisioned K/OPc) against the real AUSF/UDM, plus NAS integrity +
ciphering** — the control-plane matched-sim crutch is gone.

### One real UE-side NAS bug this gate found + fixed
First attempt: Open5GS rejected our Registration Request with 5GMM cause #95
("semantically incorrect message"), logging `Non cleartext IEs is included
[0xc]`. Per **TS 24.501 §4.4.6** the initial *unprotected* Registration Request
must carry **cleartext IEs only**; our UE sent non-cleartext IEs (Requested
NSSAI, …) in the clear. Our lenient sim AMF had accepted it. Fixed in
`nextgsim-ue/src/nas/mm/orchestrator.rs` (`build_and_send_registration`): the
unprotected initial message now carries cleartext IEs only (reg type, ngKSI, 5GS
mobile identity, UE security capability, NID); the full request (all IEs) was
already replayed in the Security Mode Complete NAS message container.

### One real UE-side RLS bug this gate found + fixed
First data-plane attempt: the UE lost its serving cell every ~13 s (`Cell lost` →
`SignalLostToConnectedCell` → new `cell_id`, `dbm=-1`), spiralling right after
registration. An RLS UDP capture on the gNB netns proved the gNB *always* acks,
but the UE's heartbeats had systematic ~2 s gaps. Root cause: **double rate-
limiting** — `send_heartbeats` was paced by a 1 s tokio `interval` *and* gated by
`should_send_heartbeats()` (a second wall-clock 1 s limiter). Under scheduling
jitter a tick landed <1 s after the last send, got gated out, and slipped the
next heartbeat to ~2 s — exceeding the 2 s cell-lost threshold. Fixed in
`nextgsim-ue/src/rls/task.rs` by letting the tokio timer be the sole pacer.
Post-fix: **0 radio-link failures, cell stable**.

**PDU session / data plane — IN PROGRESS (blocked one step deeper, in the core
N2/SM signalling).** With the cell stable, registration is rock-solid and the
**real SMF allocates a UE IP** (`UE SUPI[...] DNN[internet] IPv4[10.45.0.6]`). But
the UE's **PDU Session Establishment Accept never arrives** (`T3580 expired …
retransmitting`). The AMF's `createSmContext` succeeds (201, IP allocated) yet it
emits **no `PDUSessionResourceSetupRequest`** to the gNB (only `InitialContext
Setup`), and retransmits hit `DUPLICATED_PDU_SESSION_ID`. The gNB *has* a
`handle_pdu_session_resource_setup` handler, so the next investigation is the
AMF↔gNB N2 PDU-session-resource-setup path (why the N2 isn't sent/acted on), then
DRB + GTP-U + `ping` UE↔DN through the real UPF. See `.context/EXIT-GATE-RUNBOOK.md`.

## Config issues already surfaced + fixed (this run)
- Open5GS 2.7 AMF requires `amf.time.t3512.value` (added to `open5gs/amf.yaml`).
- The gNB binary built in `rust:latest` (Debian trixie, glibc 2.38) needs a
  `debian:trixie-slim` runtime base, not bookworm (glibc 2.36).

## For full reg + data plane (milestones 3-4)
Add Open5GS ausf/udm/udr/pcf/smf/upf services + a provisioned subscriber (Mongo)
matching the sim UE's IMSI/K/OPc, and bring up the sim UE. See
`.context/EXIT-GATE-RUNBOOK.md`.
