#!/usr/bin/env bash
# lte-ue-detach: make the running LTE UE detach (TS 24.301 §5.5.2.2, #365) and
# wait for it. nextgcore's EPC E2E detach stage (EPC_DETACH=1) runs it.
#
# Run INSIDE the lte-ue container (docker exec), after lte-ue has attached:
#   lte-ue-detach [--switch-off] [--timeout SECS]
#
# The UE sends an integrity-protected DETACH REQUEST ("EPS detach"; with
# --switch-off, the switch-off bit set, and no DETACH ACCEPT follows). It is
# EMM-DEREGISTERED after the DETACH ACCEPT, and its TUN interface goes down.
#
# Exit status: 0 once the UE is detached; 1 if it could not be reached or did
# not detach within the timeout (default 60 s).
set -euo pipefail
exec lte-ue detach "$@"
