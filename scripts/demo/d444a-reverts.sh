#!/usr/bin/env bash
# scripts/demo/d444a-reverts.sh: the D444A revert arms nextgsim can run on its
# own (nextgsim#477).
#
# Each arm applies scripts/demo/reverts/<arm>.patch through revert-run.sh and
# passes only if the named test then FAILS (and the tree is restored after):
#
#   477-ambr-event   #454's upward SessionParametersChanged is dropped (NAS still
#                    applies the AMBR and sends COMPLETE)
#                    -> a_session_parameters_change_is_forwarded_to_the_app_task
#   477-app-forward  the App task does not pass the event to the semantic codec
#                    -> a_network_session_ambr_change_cuts_the_stream_to_keep_40
#   477-keep-plan    the stream ignores the AMBR in mode semantic
#                    -> semantic_mode_follows_a_throttle_to_keep_40
#
# The other arms are config or core: `follow_session_ambr: false` and the
# routing arm (no `device:`) are config (the EKS scenario runs them), and the
# smfd step 3b patch-out is nextgcore's.
#
# Docker-free: each arm rebuilds and runs one test. ARMS="a b" runs a subset.
# Exit 0 when every arm fails its named test as it must.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../.."
declare -A test_of=(
    [477-ambr-event]="-p nextgsim-ue --bin nr-ue a_session_parameters_change_is_forwarded_to_the_app_task"
    [477-app-forward]="-p nextgsim-ue --features demo-d444a-ambr-adapt --test d444a_stream a_network_session_ambr_change_cuts_the_stream_to_keep_40"
    [477-keep-plan]="-p nextgsim-ue --features demo-d444a-ambr-adapt --lib semantic_mode_follows_a_throttle_to_keep_40"
)
arms="${ARMS:-477-ambr-event 477-app-forward 477-keep-plan}"
for arm in $arms; do
    args="${test_of[$arm]:?unknown arm $arm}"
    name="${args##* }"
    echo "[demo] revert arm $arm: $name must fail"
    # shellcheck disable=SC2086
    "$here/revert-run.sh" --expect "$name ... FAILED" "$here/reverts/$arm.patch" -- \
        cargo test $args
done
echo "[demo] every D444A revert arm failed its named test"
