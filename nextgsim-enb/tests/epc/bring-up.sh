#!/usr/bin/env bash
# Start the nextgcore EPC that `tests/mmed_interop.rs`'s
# `lte_ue_attaches_through_the_nextgcore_epc` attaches through, on loopback:
#
#   MongoDB      127.0.0.1:27118   (docker; the test subscriber below)
#   hssd         127.0.0.8         S6a, Diameter over TCP
#   sgwcd        127.0.0.3         S11 / PFCP to sgwud / S5-C to smfd
#   sgwud        127.0.0.6         PFCP / GTP-U
#   smfd (PGW-C) 127.0.0.4         S5-C / PFCP to upfd
#   upfd (PGW-U) 127.0.0.7         PFCP / GTP-U (no dataplane)
#
# The test itself spawns mmed on 127.0.0.5 (S1-MME 36412, S11 2123).
#
#   NEXTGCORE_BIN=nextgcore/src/target/debug tests/epc/bring-up.sh [up|down]
#   NEXTGCORE_EPC=1 NEXTGCORE_MMED=$NEXTGCORE_BIN/nextgcore-mmed \
#     cargo test -p nextgsim-enb --features kernel-sctp --test mmed_interop \
#     lte_ue_attaches_through_the_nextgcore_epc -- --ignored --nocapture
#
# hssd answers every AIR with 5012 until nextgcore #478 (it never opens the
# subscriber DB) is fixed; mmed's attach then stops at #479 and #480.
#
# sgwcd and sgwud take their addresses from the environment only (nextgcore
# #482), hence the variables below rather than YAML.
set -euo pipefail

DIR=${EPC_DIR:-/tmp/nextgsim-epc}
MONGO=nextgsim-epc-mongo

down() {
    if [ -d "$DIR" ]; then
        for p in "$DIR"/*.pid; do
            if [ -e "$p" ]; then
                kill "$(cat "$p")" 2>/dev/null || true
                rm -f "$p"
            fi
        done
    fi
    docker rm -f "$MONGO" >/dev/null 2>&1 || true
    sleep 1 # let the NFs release their ports
}

if [ "${1:-up}" = down ]; then
    down
    exit 0
fi

BIN=${NEXTGCORE_BIN:?set NEXTGCORE_BIN to the nextgcore target/debug directory}
down
mkdir -p "$DIR"
cd "$DIR"

docker run -d --name "$MONGO" -p 127.0.0.1:27118:27017 mongo:6 >/dev/null
for _ in $(seq 30); do
    docker exec "$MONGO" mongosh --quiet --eval 'db.runCommand({ping:1})' >/dev/null 2>&1 && break
    sleep 1
done
# The test subscriber of nextgcore's k8s/manifests/mongodb-init.yaml: TS 35.208
# K / OPc, AMF with the separation bit (TS 33.401 §6.1.1), APN "internet".
docker exec -i "$MONGO" mongosh --quiet >/dev/null <<'JS'
db = db.getSiblingDB("nextgcore");
db.subscribers.deleteMany({ imsi: "999700000000001" });
db.subscribers.insertOne({
  imsi: "999700000000001",
  msisdn: ["821000000001"],
  security: { k: "465B5CE8B199B49FAA5F0A2EE238A6BC", opc: "E8ED289DEBA952E4283B54E88E6D834D",
              amf: "8000", sqn: NumberLong(32) },
  ambr: { uplink: { value: 1, unit: 3 }, downlink: { value: 1, unit: 3 } },
  slice: [{ sst: 1, default_indicator: true, session: [{ name: "internet", type: 3,
      qos: { index: 9, arp: { priority_level: 8, pre_emption_capability: 1,
                              pre_emption_vulnerability: 1 } },
      ambr: { uplink: { value: 1, unit: 3 }, downlink: { value: 1, unit: 3 } } }] }],
  access_restriction_data: 32, subscriber_status: 0, network_access_mode: 0,
  subscribed_rau_tau_timer: 12
});
JS

cat > hss.conf <<CONF
Identity = "hss.localdomain";
Realm = "localdomain";
No_IPv6;
No_SCTP;
ListenOn = "127.0.0.8";
NoRelay;
ConnectPeer = "mme.localdomain" { ConnectTo = "127.0.0.5"; No_TLS; };
CONF
cat > hss.yaml <<CONF
db_uri: mongodb://127.0.0.1:27118/nextgcore
logger: { file: { path: $DIR/hss.log }, level: info }
hss:
  freeDiameter: $DIR/hss.conf
  diameter: { identity: hss.localdomain, realm: localdomain, addr: 127.0.0.8, port: 3868 }
  metrics: { server: [ { address: 127.0.0.8, port: 9090 } ] }
CONF
cat > pgwc.yaml <<CONF
logger: { level: info }
smf:
  sbi: { server: [ { address: 127.0.0.4, port: 7777 } ] }
  gtpc: { server: [ { address: 127.0.0.4, port: 2123 } ] }
  pfcp:
    server: [ { address: 127.0.0.4, port: 8805 } ]
    client: { upf: [ { address: 127.0.0.7 } ] }
  gtpu: { server: [ { address: 127.0.0.4 } ] }
  metrics: { server: [ { address: 127.0.0.4, port: 9090 } ] }
  session: [ { subnet: 10.45.0.0/16, gateway: 10.45.0.1 } ]
  dns: [ 8.8.8.8 ]
  mtu: 1400
  tls: { enabled: false }
CONF
cat > pgwu.yaml <<CONF
logger: { level: info }
upf:
  pfcp: { server: [ { address: 127.0.0.7, port: 8805 } ] }
  gtpu: { server: [ { address: 127.0.0.7, port: 2152 } ] }
  session: [ { subnet: 10.45.0.0/16, gateway: 10.45.0.1 } ]
  metrics: { server: [ { address: 127.0.0.7, port: 9090 } ] }
CONF
cat > sgwc.yaml <<CONF
logger: { level: info }
sgwc:
  metrics: { server: [ { address: 127.0.0.3, port: 9090 } ] }
CONF
cat > sgwu.yaml <<CONF
logger: { level: info }
sgwu:
  metrics: { server: [ { address: 127.0.0.6, port: 9090 } ] }
CONF

start() { # name, then the command
    local name=$1
    shift
    "$@" >"$DIR/$name.out" 2>&1 &
    echo $! >"$DIR/$name.pid"
}

start hssd "$BIN/nextgcore-hssd" -c hss.yaml --diameter-config "$DIR/hss.conf" \
    --db-uri mongodb://127.0.0.1:27118/nextgcore
start upfd "$BIN/nextgcore-upfd" -c pgwu.yaml --pfcp-addr 127.0.0.7 --gtpu-addr 127.0.0.7 \
    --no-dataplane
SMF_PFCP_ADDR=127.0.0.4 UPF_PFCP_ADDR=127.0.0.7 NEXTGCORE_SBI_PROFILE=dev \
    start smfd "$BIN/nextgcore-smfd" -c pgwc.yaml
PFCP_BIND_ADDR=127.0.0.6:8805 PFCP_NODE_IP=127.0.0.6 SGWU_GTPU_BIND=127.0.0.6:2152 \
    SGWU_METRICS_PORT=19096 start sgwud "$BIN/nextgcore-sgwud" -c sgwu.yaml
SGWC_S11_BIND=127.0.0.3:2123 SGWC_PFCP_BIND_ADDR=127.0.0.3:8805 SGWC_PFCP_NODE_IP=127.0.0.3 \
    SGWC_SGWU_ADDR=127.0.0.6:8805 SGWC_GTPU_ADVERTISE=127.0.0.6 SGWC_PGW_S5C=127.0.0.4 \
    SGWC_RESTART_COUNTER_FILE="$DIR/sgwc-restart-counter" \
    start sgwcd "$BIN/nextgcore-sgwcd" -c sgwc.yaml

sleep 2
for p in "$DIR"/*.pid; do
    kill -0 "$(cat "$p")" 2>/dev/null || {
        echo "$(basename "$p" .pid) exited; see $DIR/$(basename "$p" .pid).out" >&2
        exit 1
    }
done
echo "EPC up in $DIR; stop it with: $0 down"
