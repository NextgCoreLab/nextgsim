# nextgsim - Pure Rust 5G UE and gNB Simulator

A pure Rust implementation of a 5G User Equipment (UE) and gNodeB (gNB) simulator, converted from [UERANSIM](https://github.com/aligungr/UERANSIM). This implementation has **zero C library dependencies**.

## Features

- **Pure Rust**: No C/C++ dependencies, fully memory-safe
- **5G-SA Support**: Complete 5G Standalone network simulation
- **Cryptography**: Milenage, SNOW3G, ZUC, AES-based NEA/NIA algorithms, ECIES
- **Protocol Support**: NAS, NGAP (ASN.1 PER), RRC (ASN.1 UPER), GTP-U, SCTP
- **Radio Simulation**: RLS protocol for UE-gNB communication over UDP
- **Async Runtime**: Built on Tokio for high-performance concurrent operations

## Workspace Structure

```
rust_src/
├── nextgsim-gnb/       # gNB binary - 5G base station simulator
├── nextgsim-ue/        # UE binary - 5G user equipment simulator
├── nextgsim-cli/       # CLI binary - control interface
├── nextgsim-common/    # Common types (PLMN, TAI, SUPI, configs)
├── nextgsim-nas/       # NAS protocol encoding/decoding
├── nextgsim-ngap/      # NGAP protocol (ASN.1 PER codec)
├── nextgsim-rrc/       # RRC protocol (ASN.1 UPER codec)
├── nextgsim-crypto/    # Cryptographic algorithms
├── nextgsim-rls/       # Radio Link Simulation protocol
├── nextgsim-gtp/       # GTP-U tunneling protocol
├── nextgsim-sctp/      # SCTP transport layer
└── tests/              # Integration tests
```

## Building

```bash
cd rust_src
cargo build --release
```

Binaries are output to `target/release/`:
- `nextgsim-gnb` - gNB simulator
- `nextgsim-ue` - UE simulator
- `nextgsim-cli` - CLI tool

## Usage

### Running gNB

```bash
./target/release/nextgsim-gnb -c ../config/open5gs-gnb.yaml
```

### Running UE

```bash
./target/release/nextgsim-ue -c ../config/open5gs-ue.yaml
```

### Using the CLI

The CLI tool (`nextgsim-cli`) connects to running gNB/UE instances for control and monitoring:

```bash
# List running nodes
./target/release/nextgsim-cli --dump

# Connect to a UE interactively
./target/release/nextgsim-cli ue1

# Execute a command directly
./target/release/nextgsim-cli ue1 -e status
```

See [docs/cli-usage.md](docs/cli-usage.md) for complete CLI documentation.

## Configuration

Configuration files use YAML format. See the `config/` directory for examples:

- `open5gs-gnb.yaml` / `open5gs-ue.yaml` - For Open5GS core
- `free5gc-gnb.yaml` / `free5gc-ue.yaml` - For Free5GC core
- `custom-gnb.yaml` / `custom-ue.yaml` - Custom configurations

### gNB Configuration Example

```yaml
nci: '0x000000010'
gnb_id_length: 32
plmn:
  mcc: '001'
  mnc: '01'
tac: 1
nssai:
  - sst: 1
amf_configs:
  - address: 127.0.0.5
    port: 38412
link_ip: 127.0.0.1
ngap_ip: 127.0.0.1
gtp_ip: 127.0.0.1
```

### UE Configuration Example

```yaml
supi: 'imsi-001010000000001'
hplmn:
  mcc: '001'
  mnc: '01'
key: '465B5CE8B199B49FAA5F0A2EE238A6BC'
op: 'E8ED289DEBA952E4283B54E88E6183CA'
op_type: OPC
amf: '8000'
gnb_search_list:
  - 127.0.0.1
sessions:
  - type: IPv4
    apn: internet
    slice:
      sst: 1
```

## Crate Descriptions

| Crate | Description |
|-------|-------------|
| `nextgsim-common` | Common types (PLMN, TAI, SUPI, GUTI), configuration structures, error types |
| `nextgsim-crypto` | Milenage (5G-AKA), SNOW3G, ZUC, NEA1/2/3, NIA1/2/3, KDF, ECIES |
| `nextgsim-nas` | NAS message encoding/decoding per 3GPP TS 24.501 |
| `nextgsim-ngap` | NGAP ASN.1 PER codec per 3GPP TS 38.413 |
| `nextgsim-rrc` | RRC ASN.1 UPER codec per 3GPP TS 38.331 |
| `nextgsim-rls` | Radio Link Simulation protocol for UE-gNB communication |
| `nextgsim-gtp` | GTP-U header encoding/decoding for user plane tunneling |
| `nextgsim-sctp` | SCTP transport using `sctp-proto` crate |
| `nextgsim-gnb` | gNB application with NGAP, RRC, GTP, RLS tasks |
| `nextgsim-ue` | UE application with NAS, RRC, RLS tasks, TUN interface |
| `nextgsim-cli` | Command-line interface for controlling running instances |

## Key Dependencies

- **tokio** - Async runtime
- **serde** / **serde_yaml** - Configuration parsing
- **tracing** - Structured logging
- **aes**, **cmac**, **sha2**, **hmac** - Cryptographic primitives
- **x25519-dalek** - Curve25519 for ECIES
- **zuc** - ZUC cipher implementation
- **sctp-proto** - Pure Rust SCTP
- **tun-rs** - TUN interface for user plane
- **asn1-codecs** - ASN.1 PER/UPER encoding

## Development

### Code Quality

```bash
# Format code
cargo fmt

# Run lints
cargo clippy

# Run tests
cargo test
```

### Logging

Set the `RUST_LOG` environment variable to control log levels:

```bash
RUST_LOG=debug ./target/release/nextgsim-ue -c config.yaml
RUST_LOG=nextgsim_nas=trace,nextgsim_ue=debug ./target/release/nextgsim-ue -c config.yaml
```

## Architecture

The simulator uses an actor-based task model with Tokio channels for message passing:

- **gNB Tasks**: App, NGAP, RRC, GTP, RLS, SCTP
- **UE Tasks**: App, NAS, RRC, RLS, TUN

State machines follow 3GPP specifications:
- UE: MM (Mobility Management), RM (Registration Management), CM (Connection Management)
- gNB: AMF connection states

## References

- [3GPP TS 24.501](https://www.3gpp.org/DynaReport/24501.htm) - NAS protocol for 5G
- [3GPP TS 38.413](https://www.3gpp.org/DynaReport/38413.htm) - NGAP specification
- [3GPP TS 38.331](https://www.3gpp.org/DynaReport/38331.htm) - RRC specification
- [3GPP TS 33.501](https://www.3gpp.org/DynaReport/33501.htm) - 5G Security architecture
- [UERANSIM](https://github.com/aligungr/UERANSIM) - Original C++ implementation

## License

GPL-3.0 - See [LICENSE](../LICENSE) for details.
