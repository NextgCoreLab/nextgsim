# Strict-Peer Exit Gate (T6.3) — direction (a): our kernel-SCTP gNB → real Open5GS AMF

Self-contained Docker setup that runs our sim gNB (kernel SCTP, `--features
kernel-sctp`) against a **real Open5GS 2.7 AMF** — the independent peer that
proves the matched-simulator crutch is gone. PLMN 999-70 (Open5GS default,
matches our gNB).

## Run
```bash
# 1. Build the kernel-SCTP gNB Linux binary (writes binaries/nr-gnb-kernel, gitignored):
cd ../..   # nextgsim root
docker run --rm -v "$PWD":/work -w /work -e CARGO_TARGET_DIR=/tmp/t rust:latest \
  bash -c 'export PATH=/usr/local/cargo/bin:$PATH; apt-get update -qq && \
    apt-get install -y -qq libsctp-dev; cargo build -p nextgsim-gnb --features kernel-sctp && \
    cp /tmp/t/debug/nr-gnb binaries/nr-gnb-kernel'
# 2. Bring up Open5GS + gNB:
cd docker/exit-gate
docker compose up -d --build
docker logs eg-gnb -f      # gNB side
docker logs eg-amf -f      # Open5GS AMF side
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

**Next (Milestone 3+):** bring up Open5GS SMF/UPF/AUSF/UDM/UDR/PCF + a
provisioned subscriber and the sim UE for Registration → PDU session → data
plane. See `.context/EXIT-GATE-RUNBOOK.md`.

## Config issues already surfaced + fixed (this run)
- Open5GS 2.7 AMF requires `amf.time.t3512.value` (added to `open5gs/amf.yaml`).
- The gNB binary built in `rust:latest` (Debian trixie, glibc 2.38) needs a
  `debian:trixie-slim` runtime base, not bookworm (glibc 2.36).

## For full reg + data plane (milestones 3-4)
Add Open5GS ausf/udm/udr/pcf/smf/upf services + a provisioned subscriber (Mongo)
matching the sim UE's IMSI/K/OPc, and bring up the sim UE. See
`.context/EXIT-GATE-RUNBOOK.md`.
