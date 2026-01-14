# nextgsim Configuration Reference

This document provides a comprehensive reference for all configuration options available in nextgsim for both gNB (gNodeB) and UE (User Equipment) simulators.

## Configuration File Format

Configuration files use YAML format. The configuration is loaded at startup and cannot be changed at runtime.

```bash
# gNB
./nextgsim-gnb -c config/gnb.yaml

# UE
./nextgsim-ue -c config/ue.yaml
```

---

## gNB Configuration

The gNB configuration defines parameters for the 5G base station simulator.

### Complete Example

```yaml
# NR Cell Identity (36-bit value, hex or decimal)
nci: '0x000000010'

# gNB ID length in bits (22-32)
gnb_id_length: 24

# Public Land Mobile Network identifier
plmn:
  mcc: 001
  mnc: 01
  long_mnc: false

# Tracking Area Code (24-bit)
tac: 1

# Network Slice Selection Assistance Information
nssai:
  - sst: 1
    sd: [0, 0, 1]
  - sst: 2

# AMF connection configurations
amf_configs:
  - address: 127.0.0.5
    port: 38412

# IP addresses for interfaces
link_ip: 127.0.0.1
ngap_ip: 127.0.0.1
gtp_ip: 127.0.0.1
gtp_advertise_ip: null

# SCTP stream handling
ignore_stream_ids: false
```

### Parameter Reference

#### Cell Identity

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `nci` | u64 | Yes | - | NR Cell Identity (36-bit value). Can be specified in hex (`'0x000000010'`) or decimal. The NCI contains both the gNB ID and Cell ID. |
| `gnb_id_length` | u8 | Yes | - | Length of the gNB ID portion within the NCI, in bits. Valid range: 22-32. The Cell ID uses the remaining bits (36 - gnb_id_length). |

The NCI is split into:
- **gNB ID**: Upper `gnb_id_length` bits
- **Cell ID**: Lower `(36 - gnb_id_length)` bits

Example: With `nci: 0x123456789` and `gnb_id_length: 24`:
- gNB ID = `0x123456` (upper 24 bits)
- Cell ID = `0x789` (lower 12 bits)

#### PLMN Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `plmn.mcc` | u16 | Yes | - | Mobile Country Code (3 digits, 000-999) |
| `plmn.mnc` | u16 | Yes | - | Mobile Network Code (2 or 3 digits) |
| `plmn.long_mnc` | bool | No | false | Set to `true` if MNC is 3 digits (e.g., MNC 001 vs MNC 01) |

#### Tracking Area

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tac` | u32 | Yes | - | Tracking Area Code (24-bit value, 0-16777215) |

#### Network Slicing (NSSAI)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `nssai` | array | Yes | - | List of supported S-NSSAI (Single Network Slice Selection Assistance Information) |
| `nssai[].sst` | u8 | Yes | - | Slice/Service Type (0-255). Common values: 1=eMBB, 2=URLLC, 3=MIoT |
| `nssai[].sd` | [u8; 3] | No | null | Slice Differentiator (24-bit). Specified as 3-byte array `[0, 0, 1]` or omitted |

#### AMF Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `amf_configs` | array | Yes | - | List of AMF (Access and Mobility Management Function) endpoints |
| `amf_configs[].address` | IpAddr | Yes | - | IP address of the AMF (IPv4 or IPv6) |
| `amf_configs[].port` | u16 | Yes | - | SCTP port of the AMF (typically 38412 per 3GPP) |

Multiple AMFs can be configured for redundancy. The gNB will attempt to connect to each in order.

#### Network Interfaces

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `link_ip` | IpAddr | Yes | - | Local IP address for RLS (Radio Link Simulation) UDP interface |
| `ngap_ip` | IpAddr | Yes | - | Local IP address for NGAP/SCTP interface (N2 interface to AMF) |
| `gtp_ip` | IpAddr | Yes | - | Local IP address for GTP-U interface (N3 interface for user plane) |
| `gtp_advertise_ip` | IpAddr | No | null | Advertised GTP IP address for NAT scenarios. If set, this IP is sent to the UPF instead of `gtp_ip` |

#### SCTP Options

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ignore_stream_ids` | bool | No | false | If `true`, ignore SCTP stream ID errors. Useful for compatibility with some core network implementations |

---

## UE Configuration

The UE configuration defines parameters for the 5G user equipment simulator.

### Complete Example

```yaml
# Subscriber identity
supi: 'imsi-001010000000001'

# SUCI protection
protection_scheme: 0
home_network_public_key_id: 1
home_network_public_key: '5a8d38864820197c3394b92613b20b91633cbd897119273bf8e4a6f4eec0a650'
routing_indicator: '0000'

# Home PLMN
hplmn:
  mcc: 001
  mnc: 01
  long_mnc: false

# Authentication credentials
key: '465B5CE8B199B49FAA5F0A2EE238A6BC'
op: 'E8ED289DEBA952E4283B54E88E6183CA'
op_type: Opc
amf: '8000'

# Device identity
imei: '356938035643803'
imei_sv: '4370816125816151'

# Security algorithms
supported_algs:
  nia1: true
  nia2: true
  nia3: true
  nea1: true
  nea2: true
  nea3: true

# gNB discovery
gnb_search_list:
  - 127.0.0.1

# PDU sessions
sessions:
  - type: IPv4
    apn: internet
    s_nssai:
      sst: 1
    is_emergency: false

# Network slicing
configured_nssai:
  slices:
    - sst: 1
      sd: [0, 0, 1]

# TUN interface
tun_name: uesimtun0
```

### Parameter Reference

#### Subscriber Identity

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `supi` | string | No | null | Subscription Permanent Identifier. Format: `imsi-<15 digits>` (e.g., `imsi-001010000000001`) |
| `imei` | string | No | null | International Mobile Equipment Identity (15 digits). Used if SUPI is not provided |
| `imei_sv` | string | No | null | IMEI Software Version (16 digits). Used if SUPI and IMEI are not provided |

At least one of `supi`, `imei`, or `imei_sv` should be provided for UE identification.

#### SUCI Protection (Privacy)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `protection_scheme` | u8 | No | 0 | SUCI protection scheme: `0` = Null scheme (no encryption), `1` = Profile A (ECIES), `2` = Profile B |
| `home_network_public_key_id` | u8 | No | 0 | Identifier for the home network public key (0-255) |
| `home_network_public_key` | string | No | "" | Home network public key for SUCI calculation (hex string). Required for protection schemes 1 and 2 |
| `routing_indicator` | string | No | null | Routing Indicator for SUCI (4 digits, e.g., `"0000"`) |

#### Home PLMN

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `hplmn.mcc` | u16 | Yes | - | Home PLMN Mobile Country Code (3 digits) |
| `hplmn.mnc` | u16 | Yes | - | Home PLMN Mobile Network Code (2 or 3 digits) |
| `hplmn.long_mnc` | bool | No | false | Set to `true` if MNC is 3 digits |

#### Authentication Credentials

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `key` | string | Yes | - | Subscriber key K (128-bit, 32 hex characters). Shared secret with the network |
| `op` | string | Yes | - | Operator key OP or OPc (128-bit, 32 hex characters) |
| `op_type` | enum | No | `Opc` | Type of operator key: `Op` (needs conversion to OPc) or `Opc` (used directly) |
| `amf` | string | No | `"8000"` | Authentication Management Field (16-bit, 4 hex characters). Default `8000` per 3GPP |

#### Security Algorithms

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `supported_algs.nia1` | bool | No | true | Support NIA1 (SNOW3G-based integrity) |
| `supported_algs.nia2` | bool | No | true | Support NIA2 (AES-based integrity) |
| `supported_algs.nia3` | bool | No | true | Support NIA3 (ZUC-based integrity) |
| `supported_algs.nea1` | bool | No | true | Support NEA1 (SNOW3G-based ciphering) |
| `supported_algs.nea2` | bool | No | true | Support NEA2 (AES-based ciphering) |
| `supported_algs.nea3` | bool | No | true | Support NEA3 (ZUC-based ciphering) |

The network selects algorithms based on UE capabilities and network policy. NIA0/NEA0 (null algorithms) are always supported.

#### gNB Discovery

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `gnb_search_list` | array | Yes | - | List of gNB IP addresses to search for via RLS heartbeats |

The UE sends heartbeat messages to all addresses in this list and connects to the gNB with the strongest signal.

#### PDU Sessions

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sessions` | array | No | [] | List of PDU sessions to establish after registration |
| `sessions[].type` | enum | No | `Ipv4` | PDU session type: `Ipv4`, `Ipv6`, `Ipv4v6`, `Unstructured`, `Ethernet` |
| `sessions[].apn` | string | No | null | Access Point Name (Data Network Name) |
| `sessions[].s_nssai` | object | No | null | S-NSSAI for the session |
| `sessions[].s_nssai.sst` | u8 | Yes | - | Slice/Service Type |
| `sessions[].s_nssai.sd` | [u8; 3] | No | null | Slice Differentiator |
| `sessions[].is_emergency` | bool | No | false | Whether this is an emergency session |

#### Network Slicing

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `configured_nssai` | object | No | empty | Configured NSSAI for this UE by HPLMN |
| `configured_nssai.slices` | array | No | [] | List of configured S-NSSAIs |
| `configured_nssai.slices[].sst` | u8 | Yes | - | Slice/Service Type |
| `configured_nssai.slices[].sd` | [u8; 3] | No | null | Slice Differentiator |

#### TUN Interface

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tun_name` | string | No | null | Name for the TUN interface (e.g., `uesimtun0`). If not specified, a name is auto-generated |

---

## CLI Options

### gNB CLI Options

```bash
nextgsim-gnb [OPTIONS] -c <CONFIG>

Options:
  -c, --config <FILE>    Path to gNB configuration file (required)
  -l, --disable-cmd      Disable CLI command interface
  -h, --help             Print help
  -V, --version          Print version
```

### UE CLI Options

```bash
nextgsim-ue [OPTIONS] -c <CONFIG>

Options:
  -c, --config <FILE>    Path to UE configuration file (required)
  -i, --imsi <IMSI>      Override IMSI from config file
  -n, --num-of-UE <NUM>  Number of UEs to spawn (1-512, default: 1)
  -t, --tempo <MS>       Delay in milliseconds between UE starts (default: 0)
  -l, --disable-cmd      Disable CLI command interface
  -r, --no-routing-config  Disable TUN routing configuration
  -h, --help             Print help
  -V, --version          Print version
```

When spawning multiple UEs (`-n`), each UE gets an incremented IMSI (last digits incremented).

---

## Example Configurations

### Open5GS Core

**gNB** (`config/open5gs-gnb.yaml`):
```yaml
nci: '0x000000010'
gnb_id_length: 32
plmn:
  mcc: 999
  mnc: 70
tac: 1
nssai:
  - sst: 1
amf_configs:
  - address: 127.0.0.5
    port: 38412
link_ip: 127.0.0.1
ngap_ip: 127.0.0.1
gtp_ip: 127.0.0.1
ignore_stream_ids: true
```

**UE** (`config/open5gs-ue.yaml`):
```yaml
supi: 'imsi-999700000000001'
hplmn:
  mcc: 999
  mnc: 70
key: '465B5CE8B199B49FAA5F0A2EE238A6BC'
op: 'E8ED289DEBA952E4283B54E88E6183CA'
op_type: Opc
amf: '8000'
gnb_search_list:
  - 127.0.0.1
sessions:
  - type: IPv4
    apn: internet
    s_nssai:
      sst: 1
configured_nssai:
  slices:
    - sst: 1
```

### Free5GC Core

**gNB** (`config/free5gc-gnb.yaml`):
```yaml
nci: '0x000000010'
gnb_id_length: 32
plmn:
  mcc: 208
  mnc: 93
tac: 1
nssai:
  - sst: 1
    sd: [0, 1, 2]
amf_configs:
  - address: 127.0.0.1
    port: 38412
link_ip: 127.0.0.1
ngap_ip: 127.0.0.1
gtp_ip: 127.0.0.1
ignore_stream_ids: true
```

**UE** (`config/free5gc-ue.yaml`):
```yaml
supi: 'imsi-208930000000001'
hplmn:
  mcc: 208
  mnc: 93
key: '8baf473f2f8fd09487cccbd7097c6862'
op: '8e27b6af0e692e750f32667a3b14605d'
op_type: Opc
amf: '8000'
gnb_search_list:
  - 127.0.0.1
sessions:
  - type: IPv4
    apn: internet
    s_nssai:
      sst: 1
      sd: [0, 1, 2]
configured_nssai:
  slices:
    - sst: 1
      sd: [0, 1, 2]
```

---

## Troubleshooting

### Common Configuration Issues

1. **PLMN Mismatch**: Ensure `plmn` (gNB) matches `hplmn` (UE) for the UE to register.

2. **NSSAI Mismatch**: The UE's `configured_nssai` must include slices supported by the gNB's `nssai`.

3. **Authentication Failure**: Verify `key`, `op`, and `op_type` match the subscriber profile in the core network.

4. **Connection Timeout**: Check that `amf_configs` addresses are reachable and the AMF is running.

5. **gNB Not Found**: Ensure `gnb_search_list` contains the correct gNB `link_ip` address.

### Logging

Enable debug logging to troubleshoot configuration issues:

```bash
RUST_LOG=debug ./nextgsim-ue -c config.yaml
RUST_LOG=nextgsim_nas=trace ./nextgsim-ue -c config.yaml
```
