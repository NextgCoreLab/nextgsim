#!/usr/bin/env bash
# scripts/demo/d446a-reverts.sh: the D446A revert arms (nextgsim#483).
#
# Each arm applies scripts/demo/reverts/<arm>.patch through revert-run.sh and
# passes only if the named test then FAILS (and the tree is restored after):
#
#   446-supersede            every submitted intent executes again (the queue is bypassed)
#                            -> a_superseded_intent_never_executes
#   446-cause                lb-1's handover keeps the radio-reason cause
#                            -> a_loaded_cell_actuates_lb_1_with_the_load_balancing_cause
#   446-inbound-arbitration  the booster admits every inbound handover
#                            -> energy_saving_mode_refuses_and_switches_off
#   446-policy-toggle        the mode is ignored, fixed priorities decide
#                            -> the_mode_not_the_priority_decides
#   446-handback-hold        no handback hold
#                            -> a_held_source_cell_is_not_handed_back
#   446-switch-off           an off-air cell still answers heartbeats
#                            -> an_off_air_cell_answers_no_heartbeat
#   446-connected-guardrail  the guardrail counts UEs with a data-carrying bearer,
#                            not connected ones
#                            -> a_connected_but_silent_ue_holds_the_switch_off
#
# Docker-free: each arm rebuilds and runs one unit test. ARMS="a b" runs a subset.
# Exit 0 when every arm fails its named test as it must.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../.."

declare -A test_of=(
    [446-supersede]="-p nextgsim-agent --lib a_superseded_intent_never_executes"
    [446-cause]="-p nextgsim-gnb --features demo-d446a-agent-conflict --lib a_loaded_cell_actuates_lb_1_with_the_load_balancing_cause"
    [446-inbound-arbitration]="-p nextgsim-gnb --features demo-d446a-agent-conflict --lib energy_saving_mode_refuses_and_switches_off"
    [446-policy-toggle]="-p nextgsim-gnb --features demo-d446a-agent-conflict --lib the_mode_not_the_priority_decides"
    [446-handback-hold]="-p nextgsim-gnb --lib a_held_source_cell_is_not_handed_back"
    [446-switch-off]="-p nextgsim-gnb --lib an_off_air_cell_answers_no_heartbeat"
    [446-connected-guardrail]="-p nextgsim-gnb --features demo-d446a-agent-conflict --lib a_connected_but_silent_ue_holds_the_switch_off"
)
arms="${ARMS:-446-supersede 446-cause 446-inbound-arbitration 446-policy-toggle 446-handback-hold 446-switch-off 446-connected-guardrail}"
for arm in $arms; do
    args="${test_of[$arm]:?unknown arm $arm}"
    name="${args##* }"
    echo "[demo] revert arm $arm: $name must fail"
    # shellcheck disable=SC2086
    "$here/revert-run.sh" --expect "$name ... FAILED" "$here/reverts/$arm.patch" -- \
        cargo test $args
done
echo "[demo] every D446A revert arm failed its named test"
