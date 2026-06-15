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

**MILESTONE 2 (NG Setup procedure) — OPEN.** The gNB sends the 60-byte NG Setup
Request over the established association, but Open5GS logs nothing after accepting
the association (no NGAP processing, no decode error, no NG Setup Response), and
the gNB waits. **Next step:** packet-capture the N2 link to disambiguate:
```bash
# does the SCTP DATA chunk (PPID 60, the NG Setup) actually reach the AMF?
docker run --rm --net container:eg-amf nicolaka/netshoot \
  tshark -i any -d 'sctp.ppi==60,ngap' -Y 'sctp || ngap' -V
```
- If the DATA chunk never arrives → an SCTP send issue in `nextgsim-sctp/src/kernel.rs`
  (e.g. `sctp_sendmsg` flags/stream, or send before COMM_UP fully settled).
- If it arrives but Open5GS drops it silently → an **NGAP encoding delta** in our
  ogs-ngap codec vs Open5GS's strict APER decoder — i.e. a real conformance bug,
  which is exactly what this gate exists to find. (Raise Open5GS log level to
  `debug` in `open5gs/amf.yaml` to see the decode path.)

## Config issues already surfaced + fixed (this run)
- Open5GS 2.7 AMF requires `amf.time.t3512.value` (added to `open5gs/amf.yaml`).
- The gNB binary built in `rust:latest` (Debian trixie, glibc 2.38) needs a
  `debian:trixie-slim` runtime base, not bookworm (glibc 2.36).

## For full reg + data plane (milestones 3-4)
Add Open5GS ausf/udm/udr/pcf/smf/upf services + a provisioned subscriber (Mongo)
matching the sim UE's IMSI/K/OPc, and bring up the sim UE. See
`.context/EXIT-GATE-RUNBOOK.md`.
