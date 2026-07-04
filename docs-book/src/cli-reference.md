# nr-cli Reference

`nr-cli` is nextgsim's control tool for inspecting and driving running UE and gNB instances, modeled on UERANSIM's CLI (`nextgsim-cli/src/main.rs`). A second binary, `nr-loadtest`, provides multi-UE load testing (`src/loadtest_main.rs`).

> **Scope note:** these commands drive the simulator's own UE/gNB processes. Behavior is validated by the project's tests and matched-simulator E2E runs, not third-party certified. Commands mentioning 6G features are non-normative research prototypes.

## Invocation

```text
nr-cli [NODE] [-d|--dump] [-e|--exec <COMMAND>]
```

| Argument / flag | Meaning |
|---|---|
| `NODE` | Node name to connect to (e.g. `ue1`, `gnb1`); validated for min/max length |
| `-d`, `--dump` | List all running nodes and exit |
| `-e`, `--exec <COMMAND>` | Run a single node command non-interactively and exit (non-zero exit code on command error) |

With a node name and no `--exec`, `nr-cli` opens an interactive prompt (`<node>> `); `help` prints command hints, `exit`/`quit` (or EOF) leaves.

### How discovery and transport work

Running gNB/UE instances register in a process-table directory (`/tmp/nextgsim.proc-table/`, see `src/proc_table.rs`). `nr-cli --dump` scans it; connecting resolves the node's command port and talks to it over a local UDP socket on `127.0.0.1` (`src/client.rs`). Entries with a mismatched nextgsim version are skipped and reported.

```console
$ nr-cli --dump
ue1
gnb1
$ nr-cli ue1 -e status
$ nr-cli gnb1          # interactive session
```

## UE node commands

Parsed in `nextgsim-ue/src/app/task.rs` (`parse_ue_cli_command`) and executed by `nextgsim-ue/src/app/cli_handler.rs`, which triggers the corresponding NAS procedures.

| Command | Description |
|---|---|
| `info` | UE information (SUPI, active PDU sessions, pending procedures) as YAML |
| `status` | RM/MM state, MM sub-state, and PDU session list as YAML |
| `ps-list` | Alias for `status` (shows PDU sessions) |
| `timers` | Active NAS session-management timers (e.g. T3580/T3582) |
| `deregister [--switch-off\|-s]` | Initiate deregistration; `--switch-off` sends switch-off cause and does not wait for a response |
| `ps-establish [--type\|-t <ipv4\|ipv6\|ipv4v6\|unstructured\|ethernet>] [--apn\|-a <apn>] [--sst\|-s <sst>]` | Establish a PDU session (default type IPv4; PSI/PTI auto-allocated) |
| `ps-release <psi>` | Release PDU session with PSI 1–15 |
| `ps-release-all` | Release all active PDU sessions |
| `emergency-register` | Initiate emergency registration (only from DEREGISTERED state) |

Preconditions are enforced: `deregister`, `ps-establish`, `ps-release`, and `ps-release-all` require the UE to be RM-REGISTERED (and MM-REGISTERED for session commands); violations return an `ERROR:` response instead of acting.

```console
$ nr-cli ue1 -e "ps-establish --type IPv4 --apn internet"
OK: Initiating PDU session establishment (PSI: 1, PTI: 1, Type: IPv4)
$ nr-cli ue1 -e "ps-release 1"
$ nr-cli ue1 -e "deregister --switch-off"
```

## gNB node commands

Parsed and handled in `nextgsim-gnb/src/app/cmd_handler.rs` (`parse_cli_command`).

| Command | Description |
|---|---|
| `info` | gNB configuration/information as YAML |
| `status` | gNB runtime status as YAML |
| `amf-list` | List connected AMFs |
| `ue-list` | List UEs attached via this gNB |
| `ue-info <ue-id>` | Details for one UE by numeric ID |
| `ue-release <ue-id>` | Release the RRC/NGAP context of a UE by numeric ID |

```console
$ nr-cli gnb1 -e ue-list
$ nr-cli gnb1 -e "ue-info 1"
$ nr-cli gnb1 -e "ue-release 1"
```

### Help text vs. accepted commands

The interactive `help` output in `main.rs` also advertises 6G-feature commands (`nwdaf-query`, `she-offload`, `isac-sense`, `fl-train`, `semantic-encode` for UEs; `nwdaf-status`, `she-status`, `isac-config`, `fl-status`, `semantic-status`, `amf-status` for gNBs). These are **not present in the UE/gNB command parsers** listed above and will return `Unknown command`; treat the tables above (grounded in the parser code) as authoritative.

## nr-loadtest

Multi-UE load-testing binary against a 5G core (`src/loadtest_main.rs`, `src/loadtest.rs`). Exits non-zero if the registration success rate is below 90%.

| Flag | Default | Description |
|---|---|---|
| `--ues <N>` | 10 | Number of UEs to simulate |
| `--rate <N>` | 5 | Registrations per second (0 = burst) |
| `--gnb-addr <IP>` | 127.0.0.100 | gNB address |
| `--amf-addr <IP>` | 127.0.0.5 | AMF address |
| `--base-imsi <IMSI>` | 999700000000001 | Base IMSI, incremented per UE |
| `--dnn <name>` | internet | DNN for PDU sessions |
| `--sst <N>` | 1 | S-NSSAI SST |
| `--duration <secs>` | 0 | Test duration limit (0 = unlimited) |
| `--skip-pdu` | off | Skip PDU session establishment |
| `--ping` | off | Ping test after PDU session setup |
| `--log-level <lvl>` | info | Log verbosity |

```console
$ nr-loadtest --ues 100 --rate 20 --amf-addr 10.0.0.5 --ping
```
