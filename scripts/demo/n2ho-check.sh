#!/usr/bin/env bash
# scripts/demo/n2ho-check.sh: one N2 handover across two real gNBs against
# nextgcore, with the ordered chain asserted on the AMF and on both gNBs (#456).
#
#   NEXTGCORE_DIR=../nextgcore scripts/demo/n2ho-check.sh
#
# Brings up nextgcore's docker/rust/docker-compose.demo-n2ho.yml (a standalone
# stack with no published port: asserted before `up`), with THIS checkout's
# config/ mounted into the gNBs and the UE, so a revert patch to config/ takes
# effect. The UE (config/ue-n2ho.yaml) registers on gnb-west, establishes a PDU
# session, then walks toward gnb-east, and west hands it over across N2.
#
# Asserted, each chain in order within its own log (TS 23.502 §4.9.1.3,
# TS 38.413 §8.4.1-8.4.3; the strings are the ones the code emits):
#   gnb-west  Sent Handover Required -> Handover Command received
#   amf       HandoverRequired -> HandoverRequest forwarded -> HandoverRequestAcknowledge
#             -> HandoverCommand sent -> HandoverNotify -> PSI handover COMPLETED at the SMF
#   gnb-east  Handover Request received -> Sent Handover Request Acknowledge
#             -> Sent Handover Notify
# then that the request went to a different association than the command, that
# the UPF moved the downlink to gnb-east (172.23.0.102), and that a ping from the
# UE crosses the target after the handover.
#
# Environment:
#   NEXTGCORE_DIR    nextgcore checkout (default: ../nextgcore beside this repo)
#   DEMO_ARTEFACTS   where logs are captured (default /tmp/demo-artefacts/n2ho)
#   N2HO_TIMEOUT     seconds to wait for the handover after the PDU session (150)
#   N2HO_KEEP=1      leave the stack running afterwards
# Exit status: 0 when every assertion holds, 1 otherwise (DEMO CHECK FAILED: ...).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/demo/lib.sh
. "$here/lib.sh"

sim="$(cd "$here/../.." && pwd)"
core="$(cd "${NEXTGCORE_DIR:-$sim/../nextgcore}" && pwd)"
export NEXTGSIM_DIR="$sim"
demo_artefacts "${DEMO_ARTEFACTS:-/tmp/demo-artefacts}/n2ho"
timeout_s="${N2HO_TIMEOUT:-150}"

demo_compose_init "$core/docker/rust/docker-compose.demo-n2ho.yml"
demo_seed "${DEMO_SEED:-}"
demo_assert_private

teardown() {
    local rc=$?
    for s in amf smf upf gnb-west gnb-east ue; do demo_logs "$s" "$DEMO_ARTEFACTS/$s.log"; done
    if [ "${N2HO_KEEP:-0}" != 1 ]; then demo_compose down -v --remove-orphans >/dev/null 2>&1 || true; fi
    exit "$rc"
}
trap teardown EXIT

demo_compose up -d
demo_wait_log gnb-west 'Received NG Setup Response' 120
demo_wait_log gnb-east 'Received NG Setup Response' 60
demo_wait_log ue 'Received Registration Accept' 120
demo_wait_log ue 'is now ACTIVE' 60
demo_info "registered on gnb-west with a PDU session; waiting up to ${timeout_s}s for the handover"

# Wait on the last link of the chain; on a timeout fall through, so the ordered
# assertion below names the first link that is missing.
demo_wait_log amf 'handover COMPLETED at the SMF' "$timeout_s" 2>/dev/null \
    || demo_info "no completed handover within ${timeout_s}s"
for s in amf gnb-west gnb-east upf; do demo_logs "$s" "$DEMO_ARTEFACTS/$s.log"; done

assert_log_order "$DEMO_ARTEFACTS/amf.log" amf \
    'HandoverRequired \(amf_ue_ngap_id=[0-9]+, ran_ue_ngap_id=[0-9]+, type=Intra5gs\)' \
    'HandoverRequest forwarded to target association [0-9]+ for UE [0-9]+' \
    'HandoverRequestAcknowledge from association [0-9]+ \(amf_ue_ngap_id=[0-9]+, admitted [1-9][0-9]* sessions\)' \
    'HandoverCommand sent to source association [0-9]+ for UE [0-9]+' \
    'HandoverNotify from association [0-9]+ \(amf_ue_ngap_id=[0-9]+, ran_ue_ngap_id=[0-9]+\)' \
    'PSI [0-9]+ handover COMPLETED at the SMF'
assert_log_order "$DEMO_ARTEFACTS/gnb-west.log" gnb-west \
    'Sent Handover Required: amf_ue_ngap_id=' \
    'Handover Command received: amf_ue_ngap_id='
assert_log_order "$DEMO_ARTEFACTS/gnb-east.log" gnb-east \
    'Handover Request received \(target gNB\): amf_ue_ngap_id=' \
    'Sent Handover Request Acknowledge: amf_ue_ngap_id=' \
    'Sent Handover Notify: amf_ue_ngap_id='

# Two gNBs, two associations: the request went to east, the command to west.
target="$(grep -oE 'HandoverRequest forwarded to target association [0-9]+' "$DEMO_ARTEFACTS/amf.log" | head -1 | grep -oE '[0-9]+$')"
source="$(grep -oE 'HandoverCommand sent to source association [0-9]+' "$DEMO_ARTEFACTS/amf.log" | head -1 | grep -oE '[0-9]+$')"
[ "$target" != "$source" ] \
    || demo_fail "[amf] the target and source association are the same ($target): not a two-gNB handover"
demo_info "[amf] source association ${source}, target association ${target}"

# The user plane followed: the UPF's downlink moved to gnb-east at COMPLETED.
assert_log_order "$DEMO_ARTEFACTS/upf.log" upf \
    'Updated data plane session: .*gNB=172\.23\.0\.100:2152' \
    'Updated data plane session: .*gNB=172\.23\.0\.102:2152'

# And it carries traffic through the target: a ping from the UE after the handover.
demo_compose exec -T ue ping -I uesimtun0 -c 10 -i 0.5 -W 2 10.45.0.1 > "$DEMO_ARTEFACTS/ping.txt" 2>&1 || true
cat "$DEMO_ARTEFACTS/ping.txt"
received="$(sed -nE 's/.* ([0-9]+) received.*/\1/p' "$DEMO_ARTEFACTS/ping.txt")"
[ "${received:-0}" -ge 9 ] \
    || demo_fail "[ue] only ${received:-0}/10 pings crossed the target gNB after the handover"

demo_info "N2 handover gnb-west -> gnb-east: every assertion holds (logs in $DEMO_ARTEFACTS)"
