# NextGSim + NextGCore Integration Guide

This document describes how to connect NextGSim (5G UE/gNB Simulator) to NextGCore (5G Core Network).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Docker Network: nextgcore-network                       │
│                           Subnet: 172.23.0.0/24                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        NextGCore 5G Core                                │ │
│  │                                                                         │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │ │
│  │  │   NRF   │  │  AUSF   │  │   UDM   │  │   UDR   │  │   PCF   │      │ │
│  │  │ .0.10   │  │ .0.11   │  │ .0.12   │  │ .0.13   │  │ .0.14   │      │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │ │
│  │                                                                         │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────────────────────┐    │ │
│  │  │  NSSF   │  │   BSF   │  │              MongoDB                 │    │ │
│  │  │ .0.15   │  │ .0.16   │  │     .0.2:27017 (exposed 27018)      │    │ │
│  │  └─────────┘  └─────────┘  └─────────────────────────────────────┘    │ │
│  │                                                                         │ │
│  │  ┌───────────────────┐              ┌───────────────────┐             │ │
│  │  │        AMF        │◄── NGAP ────►│        SMF        │             │ │
│  │  │  .0.17:38412/sctp │              │  .0.18:8805/udp   │             │ │
│  │  │  (N2 Interface)   │              │  (N4 Interface)   │             │ │
│  │  └───────────────────┘              └─────────┬─────────┘             │ │
│  │                                               │ PFCP                   │ │
│  │                                     ┌─────────▼─────────┐             │ │
│  │                                     │        UPF        │             │ │
│  │                                     │  .0.19:2152/udp   │             │ │
│  │                                     │  (N3 Interface)   │             │ │
│  │                                     │  UE Pool: 10.45.0.0/16         │ │
│  │                                     └───────────────────┘             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        NextGSim Simulator                               │ │
│  │                                                                         │ │
│  │  ┌───────────────────┐              ┌───────────────────┐             │ │
│  │  │        gNB        │◄─── RLS ────►│         UE        │             │ │
│  │  │    172.23.0.100   │   (Radio)    │    172.23.0.101   │             │ │
│  │  │                   │              │                   │             │ │
│  │  │  • NGAP → AMF     │              │  • NAS signaling  │             │ │
│  │  │  • GTP-U → UPF    │              │  • PDU sessions   │             │ │
│  │  │  • RRC handling   │              │  • TUN interface  │             │ │
│  │  └───────────────────┘              └───────────────────┘             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
| AMF Address | 172.23.0.5:38412 | NGAP/SCTP endpoint (userspace SCTP over UDP) |
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
   docker network inspect nextgcore-network
   ```

3. **Subscriber Provisioned**: Add test subscriber to MongoDB
   ```bash
   docker exec -i nextgcore-mongodb mongosh open5gs < scripts/add-subscriber.js
   ```

## Quick Start
> **Fastest path (recommended):** the entire matched-sim E2E — our gNB+UE against
> our 22-NF 5GC, full registration + PDU session + data-plane ping (UE gets
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
# Check gNB is connected to AMF
docker exec nextgsim-gnb nr-cli gnb status

# Check UE registration status
docker exec nextgsim-ue nr-cli ue status

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

Beyond the feature configs above, URSP/UE-Policy delivery is now exercised on the
baseline path: PCF delivers URSP rules via the TS 24.501 Annex D UPDP codec over
the N1 chain, and the UE consumes them in `nextgsim-ue` (NAS MM UE-policy
handling in `nas/mm/orchestrator.rs`). SUCI de-concealment (SIDF) is performed and
logged at the core's UDM per TS 33.501 §6.12; UDR only ever sees the SUPI.
| `docker-compose.yaml` | Container orchestration |
| `scripts/add-subscriber.js` | MongoDB subscriber provisioning |

## Metrics & Monitoring

NextGCore exposes Prometheus metrics:

| Component | Metrics Endpoint |
|-----------|-----------------|
| AMF | http://localhost:9091/metrics |
| SMF | http://localhost:9092/metrics |
| PCF | http://localhost:9093/metrics |
| UPF | http://localhost:9094/metrics |

## Development Notes

### Running Locally (without Docker)

For development, you can run nextgsim binaries directly:

```bash
# Build
cargo build --release

# Run gNB
./target/release/nextgsim-gnb -c config/gnb.yaml

# Run UE (requires root for TUN)
sudo ./target/release/nextgsim-ue -c config/ue.yaml
```

### Debugging SCTP Issues

Since Docker uses userspace SCTP (over UDP), standard SCTP tools may not work. Use:

```bash
# Capture NGAP traffic
docker exec nextgcore-amf tcpdump -i eth0 port 38412 -w /tmp/ngap.pcap

# Copy and analyze with Wireshark
docker cp nextgcore-amf:/tmp/ngap.pcap ./ngap.pcap
```

## References

- [3GPP TS 38.413](https://www.3gpp.org/ftp/Specs/archive/38_series/38.413/) - NGAP Specification
- [3GPP TS 24.501](https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/) - 5G NAS Specification
- [Open5GS Documentation](https://open5gs.org/open5gs/docs/)
- [UERANSIM](https://github.com/aligungr/UERANSIM) - Original C++ implementation
