#!/usr/bin/env bash
# scripts/demo/lib.sh: the shared check library of the demo program (#456, #458).
#
# Source it from a demo check:
#   . "$(dirname "$0")/lib.sh"
#
# What it gives a check:
#   demo_seed [SEED]           seeded runs: export DEMO_SEED (given, $DEMO_SEED, or
#                              fresh) and record it, so a run can be repeated.
#   demo_compose_init FILE     run every compose call below against FILE, under
#                              project $DEMO_PROJECT (default: the file's own name:).
#   demo_assert_private        fail if the stack publishes a host port or uses host
#                              networking (`docker compose config`, the model `up` runs).
#   demo_logs SERVICE OUT      capture a service's log to a FILE.
#   demo_wait_log SERVICE ERE SECS
#                              poll until SERVICE's log matches ERE, or fail.
#   assert_log_order FILE LABEL ERE...
#                              every ERE matches FILE, each on a later line than the
#                              previous one. Prints the matched chain; on failure
#                              names the first missing or out-of-order step.
#   demo_artefacts DIR         where captured logs go ($DEMO_ARTEFACTS, default
#                              /tmp/demo-artefacts).
#
# Logs are always captured to a file and the file is grepped. Never
# `docker compose logs | grep -q`: under pipefail, grep -q's early exit SIGPIPEs
# the producer and turns a real match into a failure.
#
# Every failure goes through demo_fail, which prints `DEMO CHECK FAILED: <why>`
# on stderr and returns 1, so revert-run.sh --expect can tell a check that failed
# for the named reason from one that failed for an unrelated one.

DEMO_ARTEFACTS="${DEMO_ARTEFACTS:-/tmp/demo-artefacts}"
DEMO_COMPOSE=(docker compose)

demo_info() { printf '[demo] %s\n' "$*"; }

demo_fail() {
    printf 'DEMO CHECK FAILED: %s\n' "$*" >&2
    if [ -n "${GITHUB_ACTIONS:-}" ]; then printf '::error::%s\n' "$*" >&2; fi
    return 1
}

demo_seed() {
    DEMO_SEED="${1:-${DEMO_SEED:-}}"
    if [ -z "$DEMO_SEED" ]; then
        DEMO_SEED="$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')"
    fi
    case "$DEMO_SEED" in
        '' | *[!0-9]*) demo_fail "DEMO_SEED must be a decimal integer, got '$DEMO_SEED'"; return 1 ;;
    esac
    export DEMO_SEED
    mkdir -p "$DEMO_ARTEFACTS"
    printf '%s\n' "$DEMO_SEED" > "$DEMO_ARTEFACTS/seed"
    demo_info "seed ${DEMO_SEED} (rerun with DEMO_SEED=${DEMO_SEED})"
}

demo_artefacts() {
    DEMO_ARTEFACTS="$1"
    mkdir -p "$DEMO_ARTEFACTS"
}

demo_compose_init() {
    local file="$1"
    [ -f "$file" ] || { demo_fail "compose file not found: $file"; return 1; }
    DEMO_COMPOSE=(docker compose -f "$file")
    if [ -n "${DEMO_PROJECT:-}" ]; then DEMO_COMPOSE+=(-p "$DEMO_PROJECT"); fi
}

demo_compose() { "${DEMO_COMPOSE[@]}" "$@"; }

# The rendered model `docker compose up` would run, so an `extends`, an override
# file or a merged `ports:` list cannot hide a publish from the check.
demo_assert_private() {
    local model hits
    model="$(demo_compose config --format json)" || { demo_fail "docker compose config failed"; return 1; }
    hits="$(printf '%s' "$model" | jq -r '
        .services | to_entries[]
        | select(((.value.ports // []) | length) > 0 or .value.network_mode == "host")
        | "\(.key): ports=\(.value.ports // [] | map(.published // "") | join(",")) network_mode=\(.value.network_mode // "")"')" \
        || { demo_fail "could not read the compose model"; return 1; }
    if [ -n "$hits" ]; then
        printf '%s\n' "$hits" >&2
        demo_fail "the stack exposes a service on the host (above); demo stacks publish no port"
        return 1
    fi
    demo_info "no published port, no host networking ($(printf '%s' "$model" | jq '.services | length') services)"
}

demo_logs() {
    demo_compose logs --no-color "$1" > "$2" 2>&1 || true
}

demo_wait_log() {
    local svc="$1" re="$2" secs="$3" f deadline
    mkdir -p "$DEMO_ARTEFACTS"
    f="$DEMO_ARTEFACTS/$svc.log"
    deadline=$(( $(date +%s) + secs ))
    while :; do
        demo_logs "$svc" "$f"
        if grep -qE -- "$re" "$f"; then return 0; fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            tail -40 "$f" >&2
            demo_fail "$svc never logged /$re/ within ${secs}s"
            return 1
        fi
        sleep 2
    done
}

assert_log_order() {
    local file="$1" label="$2"
    shift 2
    local total=$# step=0 after=0 re hit line
    [ -f "$file" ] || { demo_fail "[$label] no log file $file"; return 1; }
    for re in "$@"; do
        step=$((step + 1))
        # The first match strictly after the previous step's line.
        # The ERE goes through the environment: `awk -v` would expand its escapes.
        hit="$(DEMO_RE="$re" awk -v after="$after" 'NR > after && $0 ~ ENVIRON["DEMO_RE"] { print NR; exit }' "$file")"
        if [ -z "$hit" ]; then
            if grep -qE -- "$re" "$file"; then
                demo_fail "[$label] step ${step}/${total} out of order: /$re/ appears only before step $((step - 1))"
            else
                demo_fail "[$label] step ${step}/${total} missing: /$re/"
            fi
            return 1
        fi
        line="$(sed -n "${hit}p" "$file")"
        printf '[%s] %d/%d %s\n' "$label" "$step" "$total" "$line"
        after="$hit"
    done
}
