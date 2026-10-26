# nextgsim Configuration Reference

This is the reference for the YAML configuration files of the four simulator
binaries: `nr-gnb`, `nr-ue`, `lte-enb` and `lte-ue`. The key names, types and
defaults below are read from the structs the files deserialize into:

| Binary | Struct | Source |
|--------|--------|--------|
| `nr-gnb` | `GnbConfig` | `nextgsim-common/src/config.rs` |
| `nr-ue` | `UeConfig` (and `SessionConfig`) | `nextgsim-common/src/config.rs` |
| `lte-enb` | `EnbConfig` | `nextgsim-enb/src/config.rs` |
| `lte-ue` | `LteUeConfig` | `nextgsim-lte-ue/src/config.rs` |

When this page and a struct disagree, the struct wins: every field carries a
doc comment with the full rationale, which this page shortens.

## Configuration File Format

Configuration files are YAML, loaded once at startup.

```bash
nr-gnb -c config/gnb.yaml
nr-ue  -c config/ue.yaml
lte-enb -c config/enb.yaml
lte-ue  -c config/lte-ue.yaml
```

Format rules that are easy to get wrong:

- **`nci` is an integer.** Write it unquoted, in hex or decimal
  (`nci: 0x000000010`). A quoted `'0x000000010'` is a string and is refused.
- **Keys are byte arrays.** `key` and `op` are 16 bytes and `amf` is 2 bytes,
  written as YAML sequences (`amf: [0x80, 0x00]`), not hex strings.
- **`supi` is a mapping**, `{ supi_type: Imsi, value: "999700000000001" }`, not an
  `imsi-...` string. (The `-i/--imsi` command-line override does take the
  `imsi-` form.)
- **Enum spellings are the Rust variant names**: `op_type: Opc` or `Op`;
  session `type: Ipv4`, `Ipv6`, `Ipv4v6`, `Unstructured` or `Ethernet`.
- **`plmn` / `hplmn` need all three keys**: `mcc`, `mnc` and `long_mnc`.
- **An S-NSSAI** is `{ sst: 1, sd: null }` or `{ sst: 1, sd: [0, 0, 1] }`.
- A key not listed here is ignored silently (no struct sets
  `deny_unknown_fields`), so a misspelt optional key has no effect.

"Required" below means the file is refused without the key. A key marked
optional takes the default shown when absent.

The shipped files in `config/` all parse and are the best starting point:

| File | Use |
|------|-----|
| `gnb.yaml`, `ue.yaml` | gNB and UE on the nextgcore Docker network (172.23.0.0/24) |
| `gnb-docker.yaml`, `ue-docker.yaml` | Mounted by `docker/docker-compose.yml` |
| `gnb-local.yaml`, `ue-local.yaml` | gNB and UE on the host (RLS over 127.0.0.1) against the Docker AMF |
| `gnb-kernel.yaml` | Kernel-SCTP N2 (`sctp_backend: kernel`) |
| `enb.yaml`, `lte-ue.yaml` | LTE eNB and UE for nextgcore's EPC network (172.24.0.0/24) |
| `features/*.yaml` | Rel-17/18 scenarios: RedCap, XR, MINT, SNPN accept/reject, UAV allow/deny |

---

## gNB Configuration

### Example

Cut down from `config/gnb.yaml`; it parses as it stands.

```yaml
nci: 0x000000010            # NR Cell Identity (36-bit)
gnb_id_length: 32           # 22..32
plmn:
  mcc: 999
  mnc: 70
  long_mnc: false
tac: 1
nssai:
  - sst: 1
    sd: null
amf_configs:
  - address: 172.23.0.5     # an IP address or a hostname
    port: 38412
sctp_backend: userspace     # or kernel
link_ip: 172.23.0.100       # RLS (radio) socket
ngap_ip: 172.23.0.100       # N2
gtp_ip: 172.23.0.100        # N3
ignore_stream_ids: false
upf_addr: 172.23.0.7
upf_port: 2152
si_broadcast_period_ms: 80
```

### Core parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `nci` | u64 | Yes | - | NR Cell Identity, at most 36 bits (validated). The upper `gnb_id_length` bits are the gNB ID, the rest the cell ID |
| `gnb_id_length` | u8 | Yes | - | gNB ID length in bits, 22..32 (validated) |
| `plmn` | object | Yes | - | `mcc`, `mnc`, `long_mnc` (all three required) |
| `tac` | u32 | Yes | - | Tracking Area Code, 1..16777215 (validated: 0 is refused) |
| `nssai` | list of S-NSSAI | Yes | - | Slices this gNB supports |
| `amf_configs` | list | Yes | - | At least one AMF (validated) |
| `amf_configs[].address` | string | Yes | - | IP address **or hostname** (for example a Kubernetes Service name). A hostname is resolved on connect and on every reconnect, so an AMF that restarts at a new address is found again |
| `amf_configs[].port` | u16 | Yes | - | SCTP port, normally 38412 |
| `amf_configs[].secondary_addresses` | list of string | No | `[]` | Extra AMF addresses for SCTP multi-homing |
| `link_ip` | IP | Yes | - | Local address of the RLS (radio) UDP socket, port 4997 |
| `ngap_ip` | IP | Yes | - | Local address for N2 |
| `gtp_ip` | IP | Yes | - | Local address for N3 GTP-U (port 2152) |
| `gtp_advertise_ip` | IP | No | none | GTP-U address advertised to the core instead of `gtp_ip` (NAT) |
| `ignore_stream_ids` | bool | Yes | - | Ignore SCTP stream-id mismatches |
| `sctp_backend` | `userspace` \| `kernel` | No | `userspace` | `userspace` is the in-process `sctp-proto` over UDP, which only nextgcore's matching backend speaks. `kernel` is real SCTP (IP protocol 132) via libsctp; it needs a Linux build with `--features kernel-sctp` and fails loudly at run time otherwise. Use it for a real AMF |
| `upf_addr` | string | No | none | UPF GTP-U address (IP or hostname). Unset puts the GTP task in loopback mode (it echoes user data back to the UE) |
| `upf_port` | u16 | No | 2152 | UPF GTP-U port |

### N2, N3 and timers

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ngap_tngreloc_overall_secs` | u64 | 5 | TNGRELOCoverall: wait for UE Context Release Command after a UE Context Release Request (TS 38.413 §8.3.3.4) |
| `ngap_tngreloc_prep_secs` | u64 | 10 | TNGRELOCprep: how long handover preparation may stay in flight (TS 38.413 §8.4.1.2) |
| `gtpu_echo_period_secs` | u64 | 0 | GTP-U Echo period for N3 path supervision (TS 29.281 §7.2.1). 0 disables the prober |
| `gtpu_echo_max_misses` | u32 | 3 | Unanswered Echo periods before the N3 path is declared down |
| `gtpu_restart_counter_path` | path | none | File that keeps the GTP-U Recovery restart counter across restarts (TS 23.007). Unset advertises a fixed 0 |
| `quic_enabled` | bool | false | Research flag. A QUIC transport exists in `nextgsim-sctp`, but the SCTP task logs a warning and stays on SCTP when this is set |

### Radio, RRC and mobility

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `si_broadcast_period_ms` | u64 | 80 | BCCH period: MIB every period, SIB1 every second period |
| `reselection` | object | see struct | Idle-mode reselection parameters broadcast in SIB2/SIB3/SIB4: `broadcast` (true), `q_hyst_db`, `t_reselection_s`, `cell_reselection_priority`, `s_intra_search_p`, `thresh_serving_low_p`, `intra_freq_neighbours`, `inter_freq_carriers` (`CellReselectionBroadcastConfig`) |
| `paging_default_cycle_frames` | u16 | 128 | Default paging cycle in radio frames (32, 64, 128 or 256), used when the AMF supplies no Paging DRX. Must match the UE's value |
| `accept_raw_nas_on_dcch` | bool | false | Accept a raw NAS PDU on UL-DCCH from a UE that never sent RRCSetupComplete. `false` is strict TS 38.331 behaviour |
| `as_security_enabled` | bool | false | Activate AS security (SRB1 PDCP integrity and ciphering, TS 33.501 §6.5). Must match the UE's value. Covers registration and RRC re-establishment (re-keyed with KgNB*); a handover target does not activate it yet (#325), so leave it off for handover scenarios |
| `dl_arfcn` | u32 | 632448 | Downlink ARFCN of this cell, bound into the KgNB* derivation of a handover and an RRC re-establishment; every gNB and UE must hold the same value (RLS carries no carrier frequency) |
| `scell_phys_cell_id` | u16 | none | `physCellId` of a secondary cell to configure on each UE (event A6) |
| `conditional_handover` | bool | false | Nominate the configured intra-frequency neighbours as conditional-handover candidates |
| `cho_a3_offset_db` | f64 | 3.0 | `a3-Offset` in dB: of the UE's A3 measurement configuration, the CHO execution condition, and the gNB's handover decision on a MeasurementReport. Signalled in 0.5 dB units (TS 38.331 §6.3.2): -15..15 dB in 0.5 dB steps |
| `cho_hysteresis_db` | f64 | 1.0 | Hysteresis in dB, for the same three |
| `sim_position` | `{x, y, z}` (metres, i32) | origin | Where the gNB stands. Each UE's received level is -1 dBm per metre of distance to its own `sim_position`; levels above -31 dBm all report as -31 (RSRP-Range), and a cell below -120 dBm is not heard |
| `handover_neighbours` | list of `{nci, tac, gnb_id_length?}` | `[]` | Other gNBs' cells a measurement-triggered handover may target. A reported `physCellId` (`nci % 1008`) is matched against them; the match names the target gNB ID (top `gnb_id_length` bits of `nci`, default this gNB's) and TAC for HANDOVER REQUIRED. A PCI matching none is not handed over to |
| `rlc_am_psis` | list of u8 | `[]` | PDU sessions whose DRB uses RLC Acknowledged Mode instead of Unacknowledged Mode. Must match the UE's value |
| `ims_voice_over_nr` | bool | false | Answer the AMF's UE Radio Capability Check with "IMS voice supported" when the UE also supports it |
| `nrppa_serving_rsrp_dbm` | i32 | -85 | SS-RSRP reported in an NRPPa E-CID measurement result (configured, not measured) |
| `nrppa_trp_ids` | list of u16 | `[1]` | TRP identities reported for NRPPa TRP Information Exchange |

### Rel-17/18 switches

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ntn_config` | object | none | Makes the cell an **NTN cell**: the gNB broadcasts `SIB19` (ephemeris, `common_ta_us`, `k_offset`, `ul_sync_validity_s`; TS 38.300 §16.4) and also sends it in an RRCReconfiguration's `dedicatedSystemInformationDelivery`. A UE with `gnss_position_ecef_m` set then applies an uplink timing advance and Doppler pre-compensation (§16.14.2.2). Absent is a terrestrial cell |
| `ntn_config.ephemeris` | object | 600 km LEO over 0°N 0°E | The satellite's ECEF `position_m` and `velocity_m_s`, broadcast as SIB19's `ephemerisInfo`. A value outside `EphemerisInfo-r17`'s range is refused and no SIB19 is broadcast |
| `ntn_config.ul_sync_validity_s` | u16 | 30 | `ntn-UlSyncValidityDuration` in seconds: one of 5..60 in steps of 5, 120, 180, 240, 900 |
| `ntn_config` (other keys) | | | `satellite_type`, `satellite_id`, `propagation_delay_us`, `common_ta_us`, `k_offset`, `cell_center_lat`, `cell_center_lon` and `cell_radius_km` are required inside the block; `earth_fixed`, `autonomous_ta` and `max_doppler_hz` are optional |
| `snpn_config` | object | none | SNPN cell: `nid` (required inside the block), `cag_ids`, `onboarding_enabled` |
| `mbs_enabled`, `prose_enabled`, `lcs_enabled`, `xr_qos_enabled`, `ai_ml_nr_enabled`, `ambient_iot_enabled`, `uav_enabled`, `ranging_enabled`, `energy_saving_enabled` | bool | false | Rel-17/18 feature flags. `energy_saving_enabled` spawns the energy-saving task |

### 6G research flags (gNB)

These spawn research-prototype tasks, and only in a gNB built with
`--features 6g`. All default to `false`.

| Parameter | Description |
|-----------|-------------|
| `she_enabled` | Service Hosting Environment task |
| `nwdaf_enabled` | NWDAF analytics task |
| `nwdaf_model_path` | ONNX trajectory-prediction model for the NWDAF task. No model ships; unset uses linear extrapolation |
| `nkef_enabled` | NKEF knowledge-exposure task |
| `isac_enabled` | ISAC sensing task |
| `isac_ekf_tracking` | Track with the Extended Kalman Filter instead of the scalar smoother |
| `isac_ekf_process_noise_accel` | EKF acceleration process noise; unset uses the crate default |
| `isac_anchors` | Anchor positions `[x, y, z]` in metres; defaults to a 100 m triangle at 10 m height |
| `agent_enabled` | AI agent framework task |
| `federated_learning_enabled` | Federated-learning aggregator task |
| `pqc` | Post-quantum settings (`enabled`, `kem_algorithm`, `sign_algorithm`, `hybrid_mode`). The primitives are real, but no NAS or RRC path consumes them yet |

---

## UE Configuration

### Example

Cut down from `config/ue.yaml`; it parses as it stands.

```yaml
supi:
  supi_type: Imsi
  value: "999700000000001"
protection_scheme: 0                 # 0 null, 1 Profile A, 2 Profile B
home_network_public_key_id: 0
home_network_public_key: []
routing_indicator: "0000"
hplmn:
  mcc: 999
  mnc: 70
  long_mnc: false
key: [0x46, 0x5B, 0x5C, 0xE8, 0xB1, 0x99, 0xB4, 0x9F, 0xAA, 0x5F, 0x0A, 0x2E, 0xE2, 0x38, 0xA6, 0xBC]
op: [0xE8, 0xED, 0x28, 0x9D, 0xEB, 0xA9, 0x52, 0xE4, 0x28, 0x3B, 0x54, 0xE8, 0x8E, 0x6D, 0x83, 0x4D]
op_type: Opc
amf: [0x80, 0x00]
imei: "356938035643803"
imei_sv: null
supported_algs:
  nia1: true
  nia2: true
  nia3: true
  nea1: true
  nea2: true
  nea3: true
gnb_search_list:
  - "172.23.0.100"                    # IPs or hostnames, port 4997 by default
sessions:
  - type: Ipv4
    s_nssai:
      sst: 1
      sd: null
    apn: "internet"
    is_emergency: false
configured_nssai:
  slices:
    - sst: 1
      sd: null
tun_name: null
```

### Core parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `supi` | `{supi_type, value}` | No | none | Subscription Permanent Identifier. `supi_type` is `Imsi` or `Nai` |
| `protection_scheme` | u8 | Yes | - | SUCI scheme: 0 null, 1 Profile A (ECIES X25519), 2 Profile B (ECIES secp256r1). Validated 0..2 |
| `home_network_public_key_id` | u8 | Yes | - | Home-network public key identifier |
| `home_network_public_key` | list of u8 | Yes | - | Home-network public key; `[]` for the null scheme. `config/ue.yaml` carries the TS 33.501 Annex C.4.3/C.4.4 test keys, commented out |
| `routing_indicator` | string | No | none | SUCI routing indicator, 1..4 decimal digits |
| `hplmn` | object | Yes | - | Home PLMN: `mcc`, `mnc`, `long_mnc` |
| `key` | 16 bytes | Yes | - | Subscriber key K. All zeros is refused |
| `op` | 16 bytes | Yes | - | OP or OPc, per `op_type` |
| `op_type` | `Op` \| `Opc` | Yes | - | `Op` is converted to OPc at start-up |
| `amf` | 2 bytes | Yes | - | Authentication Management Field, normally `[0x80, 0x00]` |
| `imei`, `imei_sv` | string | No | none | Device identities |
| `supported_algs` | object | Yes | - | `nia1`, `nia2`, `nia3`, `nea1`, `nea2`, `nea3` (all six required) |
| `gnb_search_list` | list of string | Yes | - | gNB RLS addresses: IPs or hostnames, optionally `:port` (default 4997). Hostnames are re-resolved periodically and on losing the cell. Must be non-empty (validated) |
| `sessions` | list | Yes | - | PDU sessions to establish after registration; may be `[]` |
| `configured_nssai` | `{slices: [...]}` | Yes | - | Configured NSSAI |
| `tun_name` | string | No | none | TUN interface name prefix: interfaces are `<tun_name><N>`, the lowest N free in the process (`uesimtun` gives `uesimtun0` first); unset gives `uesimtun<N>` |

#### PDU sessions (`sessions[]`)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `type` | enum | Yes | - | `Ipv4`, `Ipv6`, `Ipv4v6`, `Unstructured`, `Ethernet` |
| `s_nssai` | S-NSSAI | No | none | Slice for the session |
| `apn` | string | No | none | DNN |
| `is_emergency` | bool | Yes | - | Emergency PDU session |
| `requested_5qi` | u8 | No | none | An XR delay-critical GBR 5QI (82..85, Rel-18) requests an XR session; with no `apn` the UE picks the XR DNN (`xr` / `xr-split` / `xr-haptic`) |

### NAS, RRC and mobility

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_file` | path | none | File keeping the TS 24.501 Annex C.1 parameters (5G-GUTI, last visited TAI, update status, native security context) across restarts. Unset disables persistence |
| `racs_store_assigned_id` | bool | true | Keep the network-assigned UE radio capability ID (RACS) from a CONFIGURATION UPDATE COMMAND and present it in later Registration Requests |
| `paging_default_cycle_frames` | u16 | 128 | Paging cycle for the UE's own paging occasion (TS 38.304 §7.1). Must match the gNB |
| `as_security_enabled` | bool | false | AS security on SRB1. Must match the gNB. Not yet usable across a handover (#325) |
| `dl_arfcn` | u32 | 632448 | Must match every gNB's (handover and re-establishment key derivation) |
| `require_broadcast_sib1` | bool | false | Require a decoded broadcast SIB1 before a cell is selectable. With `false` a detected cell is selectable at once on a simulated SIB1, but an RRC re-establishment still waits for the selected cell's broadcast SIB1 (its shortMAC-I covers the broadcast cell identity, TS 38.331 §5.3.7.3-4): against a gNB that never broadcasts SIB1 the request is not sent, T311 expires and the UE recovers through RRC_IDLE and a new RRCSetup |
| `rlc_am_psis` | list of u8 | `[]` | PDU sessions whose DRB uses RLC AM. Must match the gNB |
| `conditional_handover` | bool | false | Store CHO candidates from RRCReconfiguration and execute one when its condition holds |
| `sim_position` | `{x, y, z}` (metres, i32) | origin | Where the UE starts; see the gNB's `sim_position` |
| `sim_movement` | `{target: {x, y, z}, speed_mps, start_after_secs}` | none | Straight-line movement from `sim_position` to `target`, starting `start_after_secs` (default 0) after start-up. Drives measurement-triggered handover; `config/ue-n2ho.yaml` is an example |
| `eutra_neighbours` | list | `[]` | E-UTRA neighbour cells the UE measures for events B1/B2 |
| `eutra_b1_threshold_dbm` | i32 | -110 | `b1-ThresholdEUTRA` of the default B1 measurement |
| `lpp_arfcn_eutra` | u32 | 0 | `arfcnEUTRA` reported in an LPP E-CID measurement (configured, not measured) |
| `plmn_user_preferred`, `plmn_operator_preferred` | list of `"mcc-mnc"` | `[]` | Preferred PLMN lists for automatic PLMN selection (TS 23.122) |
| `plmn_hp_search_interval_secs` | u32 | none | Higher-priority PLMN search timer T; clamped to 6 min..8 h |
| `ursp_evaluation` | bool | false | Evaluate URSP rules (TS 24.526 §5.2) to pick DNN, S-NSSAI, SSC mode and session type. Off: sessions come from `sessions` only |
| `ursp_rules` | list | `[]` | Locally configured URSP rules |
| `ursp_os_id`, `ursp_os_app_id` | string | none | The OS Id / OS App Id the UE reports to URSP evaluation (the simulator detects no applications) |
| `gnss_position_ecef_m` | `[f64; 3]` | none | **NTN.** The UE's GNSS position in ECEF metres. With the cell's SIB19 ephemeris it is what the timing advance and Doppler pre-compensation are computed from. Unset: no pre-compensation, logged |
| `ntn_uplink_carrier_hz` | f64 | 2e9 | **NTN.** Uplink carrier the Doppler pre-compensation is scaled by |

### Rel-17/18 Feature Configuration (UE)

These optional fields drive the Rel-17/18 features integrated end to end with
nextgcore. Each is inert unless set.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `redcap` | bool | false | **RedCap** (Rel-17). The UE signals reduced capability in the Registration Request; the AMF caps UE-AMBR and the SMF reduces session-AMBR (nextgcore knobs `REDCAP_SESS_AMBR_DL_BPS` / `REDCAP_SESS_AMBR_UL_BPS`) |
| `redcap_r18` | object | none | Rel-18 RedCap extensions |
| `snpn_config` | object | none | **SNPN** (TS 23.501 §5.30). The UE includes the NID in the Registration Request and only selects a cell broadcasting a matching NID |
| `snpn_config.nid` | string | - | Network Identifier (required inside the block), for example `"7AB01234567"`. The AMF accepts it only if it is in `AMF_SNPN_ALLOWED_NIDS` (or that variable is unset), otherwise rejects with 5GMM cause #75 |
| `snpn_config.cag_ids` | list of u32 | `[]` | Closed Access Group IDs |
| `snpn_config.onboarding_enabled` | bool | false | Request SNPN onboarding |
| `mint_config` | object | none | **MINT** / disaster roaming (TS 23.761). Each subscription runs as its own UE instance: its own RLS identity, RRC connection, NAS and `uesimtun<N>` interface, so the gNB and core see one UE per SUPI (#419). The instances share a netns; use loose reverse-path filtering (`net.ipv4.conf.all.rp_filter=2`) when their addresses share a subnet |
| `mint_config.enabled` | bool | false | Run one UE instance per subscription |
| `mint_config.secondary_supis` | list of string | `[]` | Secondary SUPIs; each must be provisioned in the core |
| `mint_config.active_subscription` | u8 | 0 | The subscription an instance is; set by the UE per instance, leave 0 |
| `mint_config.simultaneous_registration` | bool | false | Run the secondary subscriptions' instances alongside the primary; without it only the primary registers |
| `mint_config.disaster_roaming` | bool | false | Set the disaster-roaming indication on secondary registrations |
| `mint_config.dnn_subscription_map` | list of `{dnn, subscription_index}` | `[]` | Which subscription's instance establishes a DNN's PDU session |
| `uav_config` | object | none | **UAV** (TS 23.256). Registers an aerial UE and periodically sends a tracking report the AMF checks against its geofence |
| `uav_config.is_aerial_ue` | bool | false | Register as an aerial UE |
| `uav_config.uav_id`, `uav_config.uss_address` | string | none | CAA-level UAV ID; USS address |
| `uav_config.max_altitude_meters` | f64 | see struct | Reported altitude. Above the AMF ceiling (`AMF_UAV_GEOFENCE`, default 120 m) the AMF revokes authorization |
| `uav_config.remote_id_enabled`, `uav_config.c2_link_required` | bool | false | Remote ID in reports; request a C2 link |
| `uav_config.uuaa_payload_hex` | string | none | Opaque UUAA payload for the service-level authentication procedure. UUAA-MM over NAS transport needs a UE built with `--features uuaa-mm` |
| `xr_config`, `ambient_iot_config`, `ranging_config`, `v2x_config`, `pin_role` | object | none | Rel-16/18 prototypes (XR, ambient IoT, ranging, V2X, PIN). Ranging also needs `--features sidelink` |
| `prose_enabled` | bool | false | Run PC5 at all (needs `--features sidelink`) |
| `prose_config` | object | default | PC5 parameters (Layer-2 ID, application code) when `prose_enabled` |

> The nextgcore env knobs (`AMF_SNPN_ALLOWED_NIDS`, `AMF_UAV_GEOFENCE`,
> `REDCAP_SESS_AMBR_DL_BPS`/`_UL_BPS`, the XR DNN to 5QI mapping) and the matching
> example configs are documented in nextgcore's `docker/rust/README.md` and in
> `config/features/`.

### 6G research flags (UE)

All default to `false`. The task also needs the matching cargo feature
(`nextgsim-she`, `nextgsim-nwdaf`, `nextgsim-isac`, `nextgsim-fl`,
`nextgsim-semantic`) in the `nr-ue` build.

| Parameter | Description |
|-----------|-------------|
| `she_enabled` | Service Hosting Environment client |
| `ai_ml_enabled` | NWDAF reporter |
| `isac_enabled` | ISAC sensor |
| `federated_learning_enabled` | Federated-learning participant |
| `semantic_comm_enabled` | Semantic codec |
| `pqc` | Post-quantum settings; the comments in `config/ue.yaml` list the accepted values |

---

## LTE eNB Configuration (`lte-enb`)

Example: `config/enb.yaml`. The eNB must be built on Linux with
`--features kernel-sctp`: S1-MME is kernel SCTP, and a build without the
feature exits with an error.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `enb_id` | u32 | Yes | - | eNB ID (decimal or `0x` hex), interpreted as `enb_id_type`'s bit width |
| `enb_id_type` | `macro` \| `home` \| `short_macro` \| `long_macro` | No | `macro` | Which `ENB-ID` alternative to send |
| `plmn` | object | Yes | - | PLMN of the Global eNB ID |
| `tac` | u16 | Yes | - | Primary TAC (used when `supported_tas` is empty) |
| `supported_tas` | list | No | `[{tac, [plmn]}]` | Explicit Supported TAs |
| `enb_name` | string | No | none | eNB name in S1 SETUP REQUEST |
| `paging_drx` | `v32` \| `v64` \| `v128` \| `v256` | No | see struct | Default Paging DRX |
| `mme_configs` | list of `{address, port}` | Yes | - | MMEs (port 36412) |
| `s1ap_ip` | IP | Yes | - | Local S1-MME address |
| `gtp_ip` | IP | No | `s1ap_ip` | S1-U GTP-U address sent in INITIAL CONTEXT SETUP RESPONSE |
| `s1_setup_retry_secs` | u64 | No | 10 | Retry delay after an S1 Setup failure without Time To Wait, or a lost association |
| `s1_setup_response_timeout_ms` | u64 | No | 5000 | Wait for S1 SETUP RESPONSE/FAILURE |
| `radio` | object | No | see struct | `link_ip`, `rls_port` (4997), `cell_id`, `freq_band`, `dl_bandwidth` (`n6`..`n100`), `q_rx_lev_min_dbm`, `t300_ms`, `si_broadcast_period_ms` |
| `security` | object | No | see struct | AS algorithm preference lists: `integrity`, `ciphering` |
| `inactivity_timer_secs` | u64 | No | none (off) | User-inactivity release: after this many seconds with no user data and no signalling for an RRC-connected UE with an S1 connection, send UE CONTEXT RELEASE REQUEST with cause user-inactivity (TS 36.413 §8.3.2). The MME's release command then releases RRC and the UE goes ECM-IDLE. 0 is refused |

## LTE UE Configuration (`lte-ue`)

Example: `config/lte-ue.yaml`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `plmn` | object | Yes | - | The PLMN NAS selected |
| `enb_search_list` | list of string | Yes | - | eNB RLS endpoints, `"ip"` or `"ip:port"` |
| `sti` | u64 | No | random | RLS sender identity |
| `position` | `[i32; 3]` | No | origin | Simulated position (signal level) |
| `subscriber` | object | No | none | USIM and attach parameters: `imsi`, `key`, `op`, `op_type`, `sqn`, `imei_sv`, `eea`, `eia`, `apn`, `pdn_type` (`ipv4`, `ipv6`, `ipv4v6`). Without it the UE camps and never attaches |
| `tun` | object | No | none | `name` (default `lteuetun<EBI>`) and `configure_routing`. Absent, no TUN is created |

### Environment overrides (LTE containers)

`lte-enb` and `lte-ue` read `NEXTGSIM_*` variables after loading the file and
before validating it, so a bad value is refused like a bad file:

| Variable | Node | Overrides |
|---|---|---|
| `NEXTGSIM_ENB_MME_ADDR` | eNB | `mme_configs` (comma-separated `ip` or `ip:port`, port 36412 by default) |
| `NEXTGSIM_ENB_IP` | eNB | `s1ap_ip` and `radio.link_ip` |
| `NEXTGSIM_ENB_GTP_IP` | eNB | `gtp_ip` |
| `NEXTGSIM_ENB_ID` | eNB | `enb_id` (decimal or `0x` hex) |
| `NEXTGSIM_ENB_INACTIVITY_TIMER_SECS` | eNB | `inactivity_timer_secs` (seconds; empty or `off` disables it) |
| `NEXTGSIM_PLMN` | both | `plmn`, written `MCC-MNC` |
| `NEXTGSIM_TAC` | eNB | `tac` |
| `NEXTGSIM_UE_ENB_ADDR` | UE | `enb_search_list` |
| `NEXTGSIM_UE_IMSI` / `_KEY` / `_OPC` / `_APN` | UE | the `subscriber`; key and OPc as 32 hex digits |
| `NEXTGSIM_UE_TUN_NAME` | UE | `tun.name` |

`nr-gnb` and `nr-ue` read no such variables.

---

## Command-line options

The flags of every binary are listed in [cli-usage.md](cli-usage.md).

---

## Troubleshooting

1. **The file is refused.** The error names the key and line. The usual causes
   are a quoted `nci`, hex-string keys, an `imsi-...` string for `supi`, a
   missing `long_mnc`, or other spellings of the enum values (`IPv4`, `OPC`).
2. **PLMN mismatch.** `plmn` (gNB) must match `hplmn` (UE) and the core.
3. **NSSAI mismatch.** The UE's `configured_nssai` must include a slice the gNB's
   `nssai` supports.
4. **Authentication failure.** `key`, `op` and `op_type` must match the
   subscriber in the core's database.
5. **No NG Setup against a real AMF.** The default `sctp_backend: userspace` is
   SCTP over UDP. Build with `--features kernel-sctp` and set
   `sctp_backend: kernel`.
6. **gNB not found.** `gnb_search_list` must contain the gNB's `link_ip`.
7. **A paired setting differs.** `as_security_enabled`, `dl_arfcn`,
   `paging_default_cycle_frames` and `rlc_am_psis` must agree between gNB and UE.

Enable debug logging with `RUST_LOG`:

```bash
RUST_LOG=debug nr-ue -c config/ue.yaml
RUST_LOG=nextgsim_nas=trace nr-ue -c config/ue.yaml
```
