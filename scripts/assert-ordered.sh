#!/usr/bin/env bash
# assert-ordered.sh FILE PATTERN...
# Succeeds when each fixed-string PATTERN occurs in FILE on a line after the
# previous PATTERN's line; otherwise names the first one missing or out of order.
set -euo pipefail
file=$1; shift
line=0
for pat in "$@"; do
  hit=$(tail -n +"$((line + 1))" "$file" | grep -nF -m1 -- "$pat" | cut -d: -f1 || true)
  if [ -z "$hit" ]; then
    echo "::error::'$pat' not found after line $line of $file"; cat "$file"; exit 1
  fi
  line=$((line + hit))
  echo "ok line $line: $pat"
done
