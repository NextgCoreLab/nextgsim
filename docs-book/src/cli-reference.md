# nr-cli Reference

This page covers the command-line binaries and `nr-cli`, the control tool for
running gNB and UE instances. Flags are taken from each binary's clap `Args`
struct; `nr-cli` commands from the gNB and UE command parsers.

## Binaries and flags

`cargo build --release` produces six binaries:

| Binary | Crate | Description |
|--------|-------|-------------|
| `nr-gnb` | `nextgsim-gnb` | 5G gNodeB simulator |
| `nr-ue` | `nextgsim-ue` | 5G UE simulator |
| `nr-cli` | `nextgsim-cli` | Control interface for running `nr-gnb` / `nr-ue` instances |
| `nr-loadtest` | `nextgsim-cli` | Multi-UE registration load test against a 5G core |
| `lte-enb` | `nextgsim-enb` | LTE eNB simulator (needs `--features kernel-sctp` on Linux to do anything) |
| `lte-ue` | `nextgsim-lte-ue` | LTE UE simulator |

---

## nr-gnb

Connects to one or more AMFs and serves UEs over RLS.

```bash
nr-gnb [OPTIONS] --config <FILE>
```

| Option | Description |
|--------|-------------|
| `-c, --config <FILE>` | gNB configuration file (required) |
| `-l, --disable-cmd` | Do not open the `nr-cli` command interface |
| `-h, --help` | Print help |
| `-V, --version` | Print version |

```bash
nr-gnb -c config/gnb.yaml
nr-gnb -c config/gnb.yaml --disable-cmd
RUST_LOG=debug nr-gnb -c config/gnb.yaml
```

---

## nr-ue

Finds a gNB over RLS, registers, and establishes the configured PDU sessions.
It creates a TUN interface per PDU session, so it needs `CAP_NET_ADMIN`.

```bash
nr-ue [OPTIONS] --config <FILE>
```

| Option | Description |
|--------|-------------|
| `-c, --config <FILE>` | UE configuration file (required) |
| `-i, --imsi <IMSI>` | Use this IMSI instead of the config's: 15 digits, optionally prefixed `imsi-` |
| `-n, --num-of-UE <NUM>` | Run NUM UEs (1..512, default 1), the IMSI incremented for each |
| `-t, --tempo <TEMPO>` | Delay in ms between UE starts (default 0) |
| `-l, --disable-cmd` | Do not open the `nr-cli` command interface |
| `-r, --no-routing-config` | Do not configure routing for the UE TUN interface. Currently only logged |
| `-h, --help` | Print help |
| `-V, --version` | Print version |

```bash
nr-ue -c config/ue.yaml
nr-ue -c config/ue.yaml -i imsi-999700000000099
nr-ue -c config/ue.yaml -n 10 -t 100
RUST_LOG=nextgsim_nas=trace nr-ue -c config/ue.yaml
```

With `-n 5` and a config SUPI of `999700000000001` the UEs are
`999700000000001` .. `999700000000005`, registered with `nr-cli` as `ue0` ..
`ue4`.

---

## lte-enb and lte-ue

```bash
lte-enb --config <FILE>
lte-ue  --config <FILE>
```

Each takes only `-c, --config <FILE>` (plus `-h` and `-V`). Their configuration,
and the `NEXTGSIM_*` environment overrides they read, are in
[configuration.md](configuration.md#lte-enb-configuration-lte-enb). Neither
registers with `nr-cli`.

The `lte-ue` container image also ships `lte-ue-ping-check <target> [count]`,
which waits for the default bearer's PDN address and asserts a 0%-loss ping
(exit 0 pass, 1 no bearer, 2 loss).

---

## nr-cli

`nr-cli` connects to running `nr-gnb` / `nr-ue` instances on the same host.
Each instance registers its node names in `/tmp/nextgsim.proc-table/`; `nr-cli`
reads that directory and talks to the node's command port over UDP on
127.0.0.1. Entries written by a different nextgsim version are skipped.

```bash
nr-cli [OPTIONS] [NODE]
```

| Argument / option | Description |
|-------------------|-------------|
| `NODE` | Node to connect to: `gnb` for a gNB, `ue0`, `ue1`, ... for UEs |
| `-d, --dump` | List the running nodes and exit |
| `-e, --exec <COMMAND>` | Run one command and exit (non-zero exit code on a command error) |
| `-h, --help` | Print help |
| `-V, --version` | Print version |

With a node and no `--exec`, `nr-cli` opens an interactive prompt; `help` lists
commands and `exit` / `quit` leaves. The interactive help also lists some 6G
commands (`nwdaf-query`, `she-offload`, ...) that the node parsers do not accept;
they return `Unknown command`.

```bash
nr-cli --dump
nr-cli ue0 -e status
nr-cli gnb                # interactive
```

In the Docker images `nr-cli` is not installed; copy `binaries/nr-cli` into the
container and run it there (nextgcore's `docker/rust/ue-flow-check.sh` does
exactly this).

### UE commands

| Command | Description |
|---------|-------------|
| `info` | UE information (SUPI, sessions, pending procedures) as YAML |
| `status` | RM, CM and MM state and the PDU sessions, as YAML |
| `ps-list` | Same as `status` |
| `timers` | Running NAS session-management timers |
| `deregister [normal\|switch-off]` | UE-originated de-registration. `normal` (the default) waits for DEREGISTRATION ACCEPT; `switch-off` (also `--switch-off`, `-s`) does not. `disable-5g` and `remove-sim` are refused |
| `ps-establish [--type\|-t <type>] [--apn\|-a <apn>] [--sst\|-s <sst>]` | Establish a PDU session (default type IPv4) |
| `ps-release <psi>` | Release the PDU session with this PSI (1..15) |
| `ps-release-all` | Release every PDU session |
| `emergency-register` | Emergency registration (only from DEREGISTERED) |

### gNB commands

| Command | Description |
|---------|-------------|
| `info` | gNB configuration as YAML |
| `status` | gNB runtime status as YAML |
| `amf-list` | Connected AMFs |
| `ue-list` | UEs served by this gNB |
| `ue-info <ue-id>` | One UE's details |
| `ue-release <ue-id>` | Release one UE's context |
| `ue-suspend <ue-id> [t380-minutes]` | Move a UE to RRC_INACTIVE, optionally with a periodic RNAU timer |
| `xn-path-switch <ue-id> <source-amf-ue-ngap-id>` | Send a Path Switch Request for a UE, as after an Xn handover |

```bash
nr-cli ue0 -e "ps-establish --type IPv4 --apn internet --sst 1"
nr-cli ue0 -e "ps-release 1"
nr-cli ue0 -e "deregister switch-off"
nr-cli gnb -e ue-list
nr-cli gnb -e "ue-suspend 1 5"
```

---

## nr-loadtest

Registers many UEs against a 5G core and reports the success rate; it exits
non-zero if fewer than 90% register.

| Option | Default | Description |
|--------|---------|-------------|
| `--ues <UES>` | 10 | Number of UEs |
| `--rate <RATE>` | 5 | Registrations per second (0 = burst) |
| `--gnb-addr <GNB_ADDR>` | 127.0.0.100 | gNB address |
| `--amf-addr <AMF_ADDR>` | 127.0.0.5 | AMF address |
| `--base-imsi <BASE_IMSI>` | 999700000000001 | First IMSI, incremented per UE |
| `--dnn <DNN>` | internet | DNN for the PDU session |
| `--sst <SST>` | 1 | S-NSSAI SST |
| `--duration <DURATION>` | 0 | Time limit in seconds (0 = none) |
| `--skip-pdu` | off | Skip PDU session establishment |
| `--ping` | off | Ping after the PDU session is up |
| `--log-level <LOG_LEVEL>` | info | Log level |

---

> **Scope note:** these commands drive the simulator's own UE and gNB processes. Their behaviour is checked by the project's tests and the matched-simulator E2E runs, not by a third party.

## Logging

All binaries log through `tracing`; set the level with `RUST_LOG`.

```bash
RUST_LOG=debug nr-ue -c config/ue.yaml
RUST_LOG=nextgsim_nas=trace,nextgsim_rrc=debug nr-ue -c config/ue.yaml
RUST_LOG=info,nextgsim_gnb=debug,nextgsim_ngap=debug nr-gnb -c config/gnb.yaml
```

---

## Quick start

1. Start a core. The nextgcore Docker stack (`../nextgcore/docker/rust`) is the
   matched peer; with any other AMF use `sctp_backend: kernel` and a
   `kernel-sctp` build.
2. Start the gNB and wait for the NG Setup Response in its log:
   `nr-gnb -c config/gnb.yaml`
3. Start the UE: `sudo nr-ue -c config/ue.yaml`. It searches for the gNB, sets
   up RRC, registers and establishes the configured PDU sessions.
4. Check it: `nr-cli ue0 -e status`.
5. Send traffic through the session: `ping -I uesimtun0 <address>`.

## Troubleshooting

### gNB does not connect to the AMF

1. Check the AMF address and port in `amf_configs`.
2. Check the SCTP backend: `userspace` only talks to nextgcore's userspace backend.
3. Check the PLMN matches the AMF's.
4. `RUST_LOG=nextgsim_sctp=debug,nextgsim_ngap=debug`.

### UE does not find the gNB

1. `gnb_search_list` must contain the gNB's `link_ip`.
2. `RUST_LOG=nextgsim_rls=debug`.

### Registration fails

1. The UE's `hplmn` must match the gNB's `plmn`.
2. `key`, `op` and `op_type` must match the subscriber in the core.
3. `RUST_LOG=nextgsim_nas=trace`.

### PDU session fails

1. The session's S-NSSAI and DNN must be configured in the core.
2. `RUST_LOG=nextgsim_ue=debug`.
