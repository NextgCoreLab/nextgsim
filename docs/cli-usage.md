# nextgsim CLI Usage

This document describes how to use the nextgsim command-line tools.

## Binaries

nextgsim provides three binaries:

| Binary | Description |
|--------|-------------|
| `nextgsim-gnb` | 5G gNodeB simulator |
| `nextgsim-ue` | 5G User Equipment simulator |
| `nextgsim-cli` | Control interface for running instances |

---

## nextgsim-gnb

The gNB simulator connects to a 5G core network (AMF) and provides radio simulation for UEs.

### Usage

```bash
nextgsim-gnb [OPTIONS] -c <CONFIG>
```

### Options

| Option | Description |
|--------|-------------|
| `-c, --config <FILE>` | Path to gNB configuration file (required) |
| `-l, --disable-cmd` | Disable CLI command interface |
| `-h, --help` | Print help information |
| `-V, --version` | Print version |

### Examples

```bash
# Start gNB with Open5GS configuration
./nextgsim-gnb -c config/open5gs-gnb.yaml

# Start gNB without CLI interface
./nextgsim-gnb -c config/gnb.yaml --disable-cmd

# Enable debug logging
RUST_LOG=debug ./nextgsim-gnb -c config/gnb.yaml
```

---

## nextgsim-ue

The UE simulator connects to a gNB via RLS (Radio Link Simulation) and performs 5G procedures.

### Usage

```bash
nextgsim-ue [OPTIONS] -c <CONFIG>
```

### Options

| Option | Description |
|--------|-------------|
| `-c, --config <FILE>` | Path to UE configuration file (required) |
| `-i, --imsi <IMSI>` | Override IMSI from config file |
| `-n, --num-of-UE <NUM>` | Number of UEs to spawn (1-512, default: 1) |
| `-t, --tempo <MS>` | Delay in milliseconds between UE starts (default: 0) |
| `-l, --disable-cmd` | Disable CLI command interface |
| `-r, --no-routing-config` | Disable TUN routing configuration |
| `-h, --help` | Print help information |
| `-V, --version` | Print version |

### Examples

```bash
# Start single UE
./nextgsim-ue -c config/open5gs-ue.yaml

# Start UE with custom IMSI
./nextgsim-ue -c config/ue.yaml -i imsi-001010000000099

# Start 10 UEs with 100ms delay between each
./nextgsim-ue -c config/ue.yaml -n 10 -t 100

# Start UE without TUN routing (useful for testing)
./nextgsim-ue -c config/ue.yaml --no-routing-config

# Enable trace logging for NAS
RUST_LOG=nextgsim_nas=trace ./nextgsim-ue -c config/ue.yaml
```

### Multi-UE Mode

When using `-n` to spawn multiple UEs:
- Each UE gets an incremented IMSI (last digits incremented)
- The `-t` tempo option adds delay between UE starts to avoid overwhelming the network
- All UEs share the same configuration except for IMSI

Example with 5 UEs starting from IMSI 001010000000001:
```bash
./nextgsim-ue -c config/ue.yaml -n 5 -t 200
# Creates UEs with IMSIs:
# - imsi-001010000000001
# - imsi-001010000000002
# - imsi-001010000000003
# - imsi-001010000000004
# - imsi-001010000000005
```

---

## nextgsim-cli

The CLI tool connects to running gNB or UE instances to send commands and query status.

### Usage

```bash
nextgsim-cli [OPTIONS] <COMMAND>
```

### Global Options

| Option | Description |
|--------|-------------|
| `-a, --address <ADDR>` | Address of the instance (default: 127.0.0.1) |
| `-p, --port <PORT>` | CLI port of the instance |
| `-h, --help` | Print help information |

### Commands

#### Status Commands

```bash
# Get UE status
nextgsim-cli -p <port> status

# Get UE information
nextgsim-cli -p <port> info

# Get UE timer status
nextgsim-cli -p <port> timers
```

#### UE Control Commands

```bash
# Deregister from network
nextgsim-cli -p <port> deregister

# Deregister with switch-off indication
nextgsim-cli -p <port> deregister --switch-off

# Establish PDU session
nextgsim-cli -p <port> ps-establish

# Establish PDU session with options
nextgsim-cli -p <port> ps-establish --type IPv4 --apn internet --sst 1

# Release specific PDU session
nextgsim-cli -p <port> ps-release <psi>

# Release all PDU sessions
nextgsim-cli -p <port> ps-release-all
```

#### gNB Control Commands

```bash
# Get gNB status
nextgsim-cli -p <port> status

# Get gNB information
nextgsim-cli -p <port> info

# List connected UEs
nextgsim-cli -p <port> ue-list

# List connected AMFs
nextgsim-cli -p <port> amf-list
```

### Examples

```bash
# Query UE status on default port
nextgsim-cli -p 9001 status

# Establish PDU session for eMBB slice
nextgsim-cli -p 9001 ps-establish --sst 1

# Release PDU session 1
nextgsim-cli -p 9001 ps-release 1

# Deregister UE
nextgsim-cli -p 9001 deregister
```

---

## Logging

nextgsim uses the `tracing` crate for structured logging. Control log levels via the `RUST_LOG` environment variable.

### Log Levels

| Level | Description |
|-------|-------------|
| `error` | Error conditions |
| `warn` | Warning conditions |
| `info` | Informational messages (default) |
| `debug` | Debug information |
| `trace` | Detailed trace information |

### Examples

```bash
# Enable debug for all crates
RUST_LOG=debug ./nextgsim-ue -c config.yaml

# Enable trace for specific crate
RUST_LOG=nextgsim_nas=trace ./nextgsim-ue -c config.yaml

# Multiple crate levels
RUST_LOG=nextgsim_nas=trace,nextgsim_rrc=debug,nextgsim_ue=info ./nextgsim-ue -c config.yaml

# Filter by module
RUST_LOG=nextgsim_nas::mm=trace ./nextgsim-ue -c config.yaml
```

### Log Output

Logs include timestamps, levels, and structured fields:

```
2026-01-02T10:15:30.123Z  INFO nextgsim_ue: Starting UE with IMSI imsi-001010000000001
2026-01-02T10:15:30.456Z DEBUG nextgsim_rls: Cell search started, searching 1 addresses
2026-01-02T10:15:30.789Z  INFO nextgsim_rls: Cell found at 127.0.0.1, signal=-65dBm
2026-01-02T10:15:31.012Z DEBUG nextgsim_nas: Sending Registration Request
2026-01-02T10:15:31.234Z TRACE nextgsim_nas: NAS PDU: 7e004179000d0199...
```

---

## Quick Start

### 1. Start the 5G Core

Start your 5G core network (Open5GS, free5GC, etc.) with appropriate configuration.

### 2. Start gNB

```bash
./nextgsim-gnb -c config/open5gs-gnb.yaml
```

Wait for "NG Setup successful" message.

### 3. Start UE

```bash
./nextgsim-ue -c config/open5gs-ue.yaml
```

The UE will automatically:
1. Search for gNB via RLS heartbeats
2. Establish RRC connection
3. Perform registration
4. Establish configured PDU sessions

### 4. Verify Connection

```bash
# Check UE status
nextgsim-cli -p 9001 status

# Should show:
# RM State: Registered
# CM State: Connected
# PDU Sessions: 1 active
```

### 5. Test Data Plane

```bash
# Ping through TUN interface
ping -I uesimtun0 8.8.8.8
```

---

## Troubleshooting

### gNB won't connect to AMF

1. Check AMF address and port in configuration
2. Verify AMF is running and accepting connections
3. Check PLMN matches between gNB and AMF
4. Enable debug logging: `RUST_LOG=nextgsim_sctp=debug,nextgsim_ngap=debug`

### UE won't find gNB

1. Verify `gnb_search_list` contains correct gNB IP
2. Check gNB `link_ip` matches search list
3. Ensure gNB is running and radio is powered on
4. Enable debug logging: `RUST_LOG=nextgsim_rls=debug`

### Registration fails

1. Verify PLMN matches between UE (`hplmn`) and gNB (`plmn`)
2. Check authentication credentials (`key`, `op`, `op_type`)
3. Verify subscriber exists in core network
4. Enable trace logging: `RUST_LOG=nextgsim_nas=trace`

### PDU session fails

1. Check S-NSSAI configuration matches network
2. Verify APN/DNN is configured in core
3. Check SMF/UPF are running
4. Enable debug logging: `RUST_LOG=nextgsim_nas::sm=debug`
