#!/usr/bin/env bash
# Usage: run_all.sh IDS_FILE ATTEMPT
# Detaches itself (nohup setsid); log: ~/flint-swebench/runs/driver.log.
# 2 instances in parallel; resumable (skips done_ markers); up to 3 passes so
# instances killed by the memory watchdog are re-run after the others.
set -u
. "$HOME/flint-swebench/env.sh"
IDS=$(readlink -f "$1"); A=$2; ST=$R/state
mkdir -p "$R/runs" "$ST"
if [ "${_FLINT_RUN_ALL_CHILD:-}" != 1 ]; then
  _FLINT_RUN_ALL_CHILD=1 nohup setsid "$0" "$IDS" "$A" \
    >> "$R/runs/driver.log" 2>&1 < /dev/null &
  echo "driver started (pid $!), log: $R/runs/driver.log"
  exit 0
fi
exec 8>"$ST/driver_att$A.lock"
flock -n 8 || { echo "$(date '+%F %T') driver att$A already running"; exit 1; }
log() { echo "$(date '+%F %T') [driver att$A] $*"; }
"$R/scripts/podman_service.sh" || { log "podman service failed"; exit 1; }
if ! pgrep -f "$R/scripts/watchdog.sh" >/dev/null; then
  nohup setsid "$R/scripts/watchdog.sh" > /dev/null 2>&1 < /dev/null &
  WD=$!
fi
log "start: $(grep -c . "$IDS") ids from $IDS"
for pass in 1 2 3; do
  pending=$(grep -v '^\s*$' "$IDS" | while read -r id; do
    [ -f "$R/runs/att$A/eval/done_$id" ] || echo "$id"; done)
  [ -z "$pending" ] && break
  log "pass $pass: $(echo "$pending" | wc -l) pending"
  echo "$pending" | xargs -P2 -I{} "$R/scripts/run_one.sh" {} "$A"
done
left=$(grep -v '^\s*$' "$IDS" | while read -r id; do
  [ -f "$R/runs/att$A/eval/done_$id" ] || echo "$id"; done | wc -l)
log "finished; $left without done_ marker"
[ -n "${WD:-}" ] && kill "$WD" 2>/dev/null
