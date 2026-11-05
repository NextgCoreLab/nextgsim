#!/usr/bin/env bash
# Run a command, retrying it a bounded number of times with exponential backoff
# (#448).
#
#   docker/retry.sh [-n ATTEMPTS] [-d FIRST_DELAY_SECONDS] [--] COMMAND [ARGS...]
#
# Defaults: 5 attempts, delays 5s, 10s, 20s, 40s between them (75s worst case);
# RETRY_ATTEMPTS and RETRY_DELAY in the environment change them, the flags win.
# Exits 0 as soon as COMMAND does. When every attempt fails it exits with
# COMMAND's LAST exit status, so a caller under `set -e` (or a workflow step)
# fails exactly as it would have without the retry. A failure is never turned
# into a success.
#
# ONLY wrap commands that are safe to repeat: a registry pull, or a build with
# no output (nothing tagged, loaded or pushed). Never a push, or any command
# whose partial run leaves a side effect behind (a half-pushed tag, a double
# publish).
set -euo pipefail

attempts="${RETRY_ATTEMPTS:-5}"
delay="${RETRY_DELAY:-5}"
while [ $# -gt 0 ]; do
  case "$1" in
    -n) attempts="${2:?-n needs a value}"; shift 2 ;;
    -d) delay="${2:?-d needs a value}"; shift 2 ;;
    --) shift; break ;;
    -*) echo "retry.sh: unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: retry.sh [-n ATTEMPTS] [-d DELAY] [--] COMMAND [ARGS...]" >&2; exit 2; }
[[ "$attempts" =~ ^[1-9][0-9]*$ ]] || { echo "retry.sh: -n must be a positive integer, got '${attempts}'" >&2; exit 2; }
[[ "$delay" =~ ^[0-9]+$ ]] || { echo "retry.sh: -d must be a non-negative integer, got '${delay}'" >&2; exit 2; }

attempt=1
while :; do
  # The command runs as an `if` condition, so `set -e` cannot abort the loop on
  # its failure; `rc` is read in the `else` branch, where `$?` is still the
  # command's own status (after `fi` it would be 0).
  if "$@"; then
    [ "$attempt" -eq 1 ] || echo "retry.sh: succeeded on attempt ${attempt}/${attempts}: $*"
    exit 0
  else
    rc=$?
  fi
  if [ "$attempt" -ge "$attempts" ]; then
    echo "::error::retry.sh: giving up after ${attempts} attempts (last exit status ${rc}): $*"
    exit "$rc"
  fi
  echo "::warning::retry.sh: attempt ${attempt}/${attempts} failed (exit ${rc}); retrying in ${delay}s: $*"
  sleep "$delay"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
done
