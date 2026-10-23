# NextGSim + NextGCore Integration Guide

This document describes how to connect NextGSim (5G UE/gNB Simulator) to NextGCore (5G Core Network).

## Architecture Overview

The nextgcore 5GC compose stack (`nextgcore/docker/rust/docker-compose.yml`,
compose network `core`, which Docker names `rust_core`, subnet 172.23.0.0/24)
puts the simulator on the same bridge as the core. The addresses this guide
relies on, from that file's `ipv4_address` entries:

| Container | Address | Interface |
|-----------|---------|-----------|
| `nextgcore-mongodb` | 172.23.0.2:27017 | subscriber database |
| `nextgcore-nrf` | 172.23.0.10:7777 | SBI |
| `nextgcore-amf` | 172.23.0.5:38412 | N2 (userspace SCTP over UDP) |
| `nextgcore-smf` | 172.23.0.4:8805 | N4 |
| `nextgcore-upf` | 172.23.0.7:2152 | N3; UE pool 10.45.0.0/16 |
| `nextgsim-gnb` | 172.23.0.100 | RLS, N2, N3 |
| `nextgsim-ue` | 172.23.0.101 | RLS; `uesimtun0` inside the container |

The other NFs in that file have their own addresses there. nextgcore's EPC
stack (`docker-compose-epc.yml`, 172.24.0.0/24) is the peer for `lte-enb` and
`lte-ue`; `config/enb.yaml` and `config/lte-ue.yaml` target it.

## Protocol Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        Control Plane                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    UE                    gNB                    AMF              │
│    ├── NAS ──────────────┼── NGAP ──────────────┤               │
│    │   (5GMM/5GSM)       │   (N2 Interface)     │               │
│    │                     │                      │               │
│    ├── RRC ──────────────┤                      │               │
│    │   (Radio Resource)  │                      │               │
│    │                     │                      │               │
│    └── RLS ──────────────┘                      │               │
│        (Simulated Radio)                        │               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                         User Plane                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    UE                    gNB                    UPF              │
│    ├── IP ───────────────┼── GTP-U ─────────────┤               │
│    │   (TUN interface)   │   (N3 Interface)     │               │
│    │   uesimtun0         │   Port 2152/UDP      │               │
│    │                     │                      │               │
│    └─────────────────────┴──────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Network Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| PLMN | 999-70 | MCC=999, MNC=70 |
| TAC | 1 | Tracking Area Code |
| S-NSSAI | SST=1 | Default network slice |
| AMF Address | 172.23.0.5:38412 | NGAP endpoint (userspace SCTP over UDP by default; `docker-compose.kernel-sctp.yml` switches N2 to kernel SCTP) |
| UPF Address | 172.23.0.7:2152 | GTP-U endpoint |
| UE IP Pool | 10.45.0.0/16 | Assigned to UEs |

### Subscriber Credentials

| Parameter | Value |
|-----------|-------|
| IMSI | 999700000000001 |
| K | 465B5CE8B199B49FAA5F0A2EE238A6BC |
| OPc | E8ED289DEBA952E4283B54E88E6D834D |
| AMF | 8000 |
| APN | internet |

## Prerequisites

1. **NextGCore Running**: Ensure the 5G core is up and healthy
   ```bash
   docker ps | grep nextgcore
   ```

2. **Network Available**: Verify the Docker network exists
   ```bash
   docker network inspect rust_core
   ```

3. **Subscriber Provisioned**: Add test subscriber to MongoDB
   ```bash
   docker exec -i nextgcore-mongodb mongosh open5gs < scripts/add-subscriber.js
   ```
   nextgcore's compose stack seeds this subscriber from its
   `docs/assets/webui/mongo-init.js` when MongoDB starts on a fresh volume, so
   this is only needed against a database that lacks it.

## Quick Start
> **Fastest path (recommended):** the entire matched-sim E2E — our gNB+UE against
> the nextgcore 5GC, full registration + PDU session + data-plane ping (UE gets
> 10.45.0.2, `uesimtun0` up, ping through the GTP-U tunnel) — is driven by one
> command from the nextgcore repo. It bundles disk preflight (hazard #269), image
> build, and the assertion run:
>
> ```bash
> cd /path/to/nextgcore/docker/rust
> ./e2e.sh                 # preflight -> build -> E2E (exit 0/1/2)
> ./e2e.sh --quick --keep  # reuse prebuilt binaries, leave the stack up
> ```
>
> See `nextgcore/docker/rust/CI.md`. The nextgsim-only compose below
> (`docker/docker-compose.yml`, gNB + UE only) is for attaching the simulator to an
> already-running core — run its `docker compose` commands from the `docker/`
> subdirectory, not the repo root.

### 1. Build NextGSim Images

```bash
cd /path/to/nextgsim/docker
docker compose build
```

### 2. Start the Simulator

```bash
docker compose up -d
```

### 3. Monitor Logs

```bash
# Watch gNB logs (NG Setup, RRC handling)
docker logs -f nextgsim-gnb

# Watch UE logs (Registration, PDU sessions)
docker logs -f nextgsim-ue

# Watch AMF logs (NGAP messages)
docker logs -f nextgcore-amf
```

### 4. Verify Connection

```bash
# nr-cli is not in the images: copy it in first
docker cp binaries/nr-cli nextgsim-gnb:/tmp/nr-cli
docker cp binaries/nr-cli nextgsim-ue:/tmp/nr-cli

# Check gNB is connected to AMF
docker exec nextgsim-gnb /tmp/nr-cli gnb -e amf-list

# Check UE registration status
docker exec nextgsim-ue /tmp/nr-cli ue0 -e status

# Ping from UE through the core
docker exec nextgsim-ue ping -I uesimtun0 10.45.0.1
```

## Troubleshooting

### NG Setup Failure

If gNB fails to connect to AMF:

1. **Check network connectivity**
   ```bash
   docker exec nextgsim-gnb ping 172.23.0.5
   ```

2. **Verify PLMN matches**
   - gNB config: `config/gnb.yaml` → plmn: mcc=999, mnc=70
   - AMF config: `nextgcore/docker/rust/configs/5gc/amf.yaml`

3. **Check AMF logs for errors**
   ```bash
   docker logs nextgcore-amf | grep -i error
   ```

### UE Registration Failure

If UE fails to register:

1. **Verify subscriber exists in MongoDB**
   ```bash
   docker exec nextgcore-mongodb mongosh open5gs --eval "db.subscribers.findOne({imsi:'999700000000001'})"
   ```

2. **Check authentication keys match**
   - UE config K/OPc must match MongoDB subscriber security.k/security.opc

3. **Check UDM/AUSF logs**
   ```bash
   docker logs nextgcore-ausf
   docker logs nextgcore-udm
   ```

### PDU Session Failure

If PDU session establishment fails:

1. **Check SMF logs**
   ```bash
   docker logs nextgcore-smf | grep -i session
   ```

2. **Verify UPF is reachable from SMF**
   ```bash
   docker exec nextgcore-smf ping 172.23.0.7
   ```

3. **Check TUN device exists in UE**
   ```bash
   docker exec nextgsim-ue ip addr show uesimtun0
   ```

## Rel-17/18 Feature Testing

Beyond the baseline registration/PDU/ping flow, the integrated Rel-17/18 features
(RedCap, XR 5QI, SNPN, MINT, UAV) are exercised end-to-end by an automated harness
that lives in nextgcore and drives nextgsim with per-feature UE/gNB configs from
`config/features/`:

```bash
# From nextgcore/docker/rust (with a built stack up, e.g. ./e2e-test.sh --keep)
./feature-e2e-test.sh                       # all feature scenarios
./feature-e2e-test.sh redcap xr snpn-accept # selected scenarios
```

Each scenario swaps the UE (and, for SNPN, the gNB + AMF) config via the
`docker-compose.features.yml` overlay and asserts the per-NF log signatures. The
matching UE-side config knobs (`redcap`, `requested_5qi`, `snpn_config`,
`mint_config`, `uav_config`) are documented in
[configuration.md](configuration.md#rel-1718-feature-configuration-ue); the
nextgcore env knobs (`AMF_SNPN_ALLOWED_NIDS`, `AMF_UAV_GEOFENCE`,
`REDCAP_SESS_AMBR_DL_BPS`/`_UL_BPS`) are documented in
`nextgcore/docker/rust/README.md`. Example configs: `config/features/*.yaml`.

## File Reference

| File | Purpose |
|------|---------|
| `config/gnb.yaml` | gNB configuration (PLMN, TAC, AMF address) |
| `config/ue.yaml` | UE configuration (IMSI, keys, sessions) |
| `config/features/*.yaml` | Per-feature UE/gNB configs for the Rel-17/18 feature E2E test |
| `config/enb.yaml`, `config/lte-ue.yaml` | LTE eNB and UE for nextgcore's EPC stack |
| `docker/docker-compose.yml` | gNB + UE containers on an existing `rust_core` network |
| `scripts/add-subscriber.js` | MongoDB subscriber provisioning |

Beyond the feature configs above, URSP/UE-Policy delivery is now exercised on the
baseline path: PCF delivers URSP rules via the TS 24.501 Annex D UPDP codec over
the N1 chain, and the UE consumes them in `nextgsim-ue` (NAS MM UE-policy
handling in `nas/mm/orchestrator.rs`).

The UE also **evaluates** those rules per TS 24.526 §5.2 — walking them in
ascending precedence, matching traffic descriptors against the detected
application, and applying the selected route selection descriptor's DNN /
S-NSSAI / SSC mode / PDU session type to the PDU session it establishes or
reuses (`nas/sm/ursp.rs`). Evaluation is **off by default** because it changes
which slice and data network a session uses; set `ursp_evaluation: true` in the
UE config to turn it on. With it off, session parameters come from the static
`sessions` config exactly as before. SUCI de-concealment (SIDF) is performed and
logged at the core's UDM per TS 33.501 §6.12; UDR only ever sees the SUPI.

## Nightly UE-flow gate and the EPC stage

nextgcore's nightly CI builds this repo from `main` and, after the baseline
registration and ping, runs `docker/rust/ue-flow-check.sh` against the real
`nr-gnb` and `nr-ue`. Each scenario passes only on a positive assertion:

| Scenario | What it drives |
|----------|----------------|
| `paging` | CM-IDLE, then downlink data: the AMF pages, the UE answers with a Service Request, every downlink ping is answered |
| `idle-uplink` | CM-IDLE, then uplink data: a UE-triggered Service Request, and the packet that triggered it is not lost |
| `deregister` | `nr-cli <ue> -e deregister`: Deregistration Accept, then a fresh registration, PDU session and ping |
| `network-rereg` | A UDM de-registration notification with `REREGISTRATION_REQUIRED`: the UE re-registers promptly and pings again |
| `all` | All of the above, with a second paging round after the Service Request |

The EPC stage of the same pipeline runs `lte-enb` and `lte-ue` against
nextgcore's MME, HSS, PCRF, SGW-C/SGW-U and PGW-C/PGW-U (`docker/rust/epc-attach-check.sh`) and
asserts a real LTE attach and a 0%-loss ping through the PDN.

This repo's own `ci.yml` runs an interop job nightly too (05:00 UTC, after
nextgcore's), which registers a real UE through this repo's gNB against
nextgcore `main`.

## Metrics & Monitoring

The nextgcore NFs serve Prometheus metrics on port 9090 of their container
address (for example the AMF at `http://172.23.0.5:9090/metrics`; see the
`metrics` block of each file in `nextgcore/docker/rust/configs/5gc/`). The
stack's Prometheus is published on `http://localhost:9090` and Grafana on
`http://localhost:3000`.

## Development Notes

### Running Locally (without Docker)

For development, you can run nextgsim binaries directly:

```bash
# Build
cargo build --release

# Run gNB on the host against the Docker AMF
./target/release/nr-gnb -c config/gnb-local.yaml

# Run UE (requires CAP_NET_ADMIN for TUN)
sudo ./target/release/nr-ue -c config/ue-local.yaml
```

### Debugging SCTP Issues

The default N2 is userspace SCTP over UDP, so SCTP dissectors do not decode it as NGAP. Capture the UDP port instead:

```bash
# Capture NGAP traffic
docker exec nextgcore-amf tcpdump -i eth0 port 38412 -w /tmp/ngap.pcap

# Copy and analyze with Wireshark
docker cp nextgcore-amf:/tmp/ngap.pcap ./ngap.pcap
```

## The nextgcore-sctp dependency

`nextgsim-sctp` takes the SCTP core it shares with the AMF from nextgcore
instead of keeping its own copy (#205): `NGAP_PPID`, the three association
defaults (`DEFAULT_NUM_STREAMS`, `DEFAULT_MAX_MESSAGE_SIZE`,
`DEFAULT_RECEIVE_BUFFER_SIZE`) and `ReceivedMessage`. It is a git dependency
**pinned by `rev`**, declared in `nextgsim-sctp/Cargo.toml`. CI needs no extra
checkout step, because cargo fetches the public repository itself. nextgcore
has no root `Cargo.toml` (its workspace root is `src/Cargo.toml`), and cargo
finds the member anyway.

**Only transport is shared.** Each repo keeps the client or server half it
actually drives: nextgcore is the SCTP server for NGAP, and this repo is the
client, including multihoming. The NGAP, NAS and NRPPa codecs stay
independent. This repo is the peer that tests nextgcore, so a misreading the
two shared would pass the cross-repo E2E (#45 found one that way).

### Bumping the rev

1. Pick a nextgcore sha that is on `main` **and** that the cross-repo E2E
   validated. Use the `validated-pair.json` artifact that nextgcore's
   `ci.yml` writes once all three heavy jobs pass. Take it from a `schedule`
   run, and check its `event` and `nextgcore_ref` fields as well as the shas:
   a manifest from a branch dispatch passes every check while naming a
   feature branch. Don't pin a branch head or an unmerged sha, because a
   rebase or a squash-merge would orphan it.
2. Replace the `rev` in `nextgsim-sctp/Cargo.toml`, then run
   `cargo update -p nextgcore-sctp` so `Cargo.lock` follows.
3. Run `cargo test --workspace --no-fail-fast`. A compile error here means
   nextgcore renamed or removed something this repo consumes. Port to the new
   name, or ask nextgcore to keep the item. Don't redefine it locally: that
   recreates the copy this dependency removed.
4. Name the old and new shas, and the manifest run they came from, in the
   commit message.

A nextgcore change does **not** reach this repo until somebody bumps the rev.
That is deliberate. It also means nextgcore's own E2E, which builds this repo,
compiles these constants from the pinned rev, not from the nextgcore commit
under test. That doesn't matter for constants. It would matter if anything
with behaviour were shared.

### The patched sctp-proto

The userspace SCTP backend runs on sctp-proto 0.6.0 patched to carry any PPID
verbatim (#220): upstream maps every non-WebRTC PPID to `Unknown`, sent as 58.
That patched crate has **one home**, nextgcore's `src/vendor/sctp-proto`
(nextgcore #467/#468; its `HOME.md` and `NEXTGSIM-PATCH.md` record the patch).
This repo takes it through `[patch.crates-io]` in the root `Cargo.toml`, as a git
dependency on nextgcore pinned by `rev` (#230), instead of vendoring a second
copy. Cargo finds the crate inside the nextgcore checkout by name even though it
is excluded from nextgcore's workspace.

Unlike the constants above, this **is** shared behaviour: the bytes on the wire
come from nextgcore's copy. So:

- The `rev` here differs from `nextgcore-sctp`'s on purpose. `nextgcore-sctp`
  stays on the E2E-validated pair (the procedure above), which predates #468.
  This `rev` only has to contain #468. Unify the two the next time
  `nextgcore-sctp` is bumped to a validated pair at or after it.
- To change the patch, change it in nextgcore, then bump this `rev` and run
  `cargo update -p sctp-proto`. `nextgsim-sctp/tests/userspace_ppid_on_wire.rs`
  is the proof that the patch is in effect: 60 and 18 on the wire, both ways.
  Without the patch the crate does not compile at all, because nextgsim-sctp
  converts the PPID to `u32`.

## References

- [3GPP TS 38.413](https://www.3gpp.org/ftp/Specs/archive/38_series/38.413/) - NGAP Specification
- [3GPP TS 24.501](https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/) - 5G NAS Specification
- [Open5GS Documentation](https://open5gs.org/open5gs/docs/)
- [UERANSIM](https://github.com/aligungr/UERANSIM) - Original C++ implementation
