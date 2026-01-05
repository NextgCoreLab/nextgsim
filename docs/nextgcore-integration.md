# NextGSim + NextGCore Integration Guide

This document describes how to connect NextGSim (5G UE/gNB Simulator) to NextGCore (5G Core Network).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Docker Network: nextgcore-network                       │
│                           Subnet: 172.22.0.0/24                              │
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
│  │  │    172.22.0.100   │   (Radio)    │    172.22.0.101   │             │ │
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
| AMF Address | 172.22.0.17:38412 | NGAP/SCTP endpoint |
| UPF Address | 172.22.0.19:2152 | GTP-U endpoint |
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

### 1. Build NextGSim Images

```bash
cd /path/to/nextgsim
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
docker exec nextgsim-ue ping -I uesimtun0 8.8.8.8
```

## Troubleshooting

### NG Setup Failure

If gNB fails to connect to AMF:

1. **Check network connectivity**
   ```bash
   docker exec nextgsim-gnb ping 172.22.0.17
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
   docker exec nextgcore-smf ping 172.22.0.19
   ```

3. **Check TUN device exists in UE**
   ```bash
   docker exec nextgsim-ue ip addr show uesimtun0
   ```

## File Reference

| File | Purpose |
|------|---------|
| `config/gnb.yaml` | gNB configuration (PLMN, TAC, AMF address) |
| `config/ue.yaml` | UE configuration (IMSI, keys, sessions) |
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
