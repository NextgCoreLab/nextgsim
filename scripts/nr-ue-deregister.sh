#!/usr/bin/env bash
# nr-ue-deregister: make a running 5G UE de-register (TS 24.501 §5.5.2.2, #365)
# and wait until it is RM-DEREGISTERED. The 5G counterpart of lte-ue-detach.
#
# Run INSIDE the nr-ue container (docker exec). Needs nr-cli in the container
# (the images ship it when binaries/nr-cli is present at build time; otherwise
# copy it in, as nextgcore's ue-flow-check.sh does):
#   nr-ue-deregister [NODE] [normal|switch-off]
# NODE defaults to the first UE `nr-cli --dump` lists; the type to "normal",
# which the AMF answers with DEREGISTRATION ACCEPT.
#
# Exit status: 0 once RM-DEREGISTERED; 1 if the command was refused or the UE
# did not de-register within NR_UE_DEREGISTER_WAIT_SECS (default 30).
set -euo pipefail

node="${1:-}"
kind="${2:-normal}"
wait_secs="${NR_UE_DEREGISTER_WAIT_SECS:-30}"
if [ -z "$node" ]; then
  node="$(nr-cli --dump | grep -m1 -E '^ue[0-9]+' || true)"
fi
if [ -z "$node" ]; then
  echo "nr-ue-deregister: no running UE found (nr-cli --dump)" >&2
  exit 1
fi
nr-cli "$node" -e "deregister $kind"
for _ in $(seq 1 "$wait_secs"); do
  if nr-cli "$node" -e status 2>/dev/null | grep -q "RM-DEREGISTERED"; then
    echo "nr-ue-deregister: $node is RM-DEREGISTERED"
    exit 0
  fi
  sleep 1
done
echo "nr-ue-deregister: $node is not RM-DEREGISTERED after ${wait_secs}s" >&2
nr-cli "$node" -e status >&2 || true
exit 1
