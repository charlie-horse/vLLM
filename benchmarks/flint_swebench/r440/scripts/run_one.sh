#!/usr/bin/env bash
# Usage: run_one.sh INSTANCE_ID ATTEMPT
#   pre-pull (global rate limit) -> agent (podman, niced, API retry/re-run)
#   -> official eval (podman socket) -> image cleanup.
# Output layout (unchanged, read by scripts/analyze.py):
#   runs/attA/{agent/ID, eval/done_ID, log/ID.{agent,eval}.log}
# done_ID may carry "infra": "pull_failed" | "api_failed".
# Exit 3 = killed by the memory watchdog (no done_ marker -> re-run later).
set -u
. "$HOME/flint-swebench/env.sh"
: "${FLINT_API_BASE:?}"
ID=$1; A=$2; OUT=$R/runs/att$A; ST=$R/state
mkdir -p "$OUT/agent" "$OUT/eval" "$OUT/log" "$ST/active"
REPORT=$OUT/eval/done_$ID
[ -f "$REPORT" ] && exit 0
LOG=$OUT/log/$ID.run.log
TAG=$(echo "$ID" | sed 's/__/_1776_/' | tr 'A-Z' 'a-z')
IMG=docker.io/swebench/sweb.eval.x86_64.$TAG:latest
PULL_GAP=240                                # >= 4 min between pulls (global)
PULL_BACKOFF=(0 120 240 480 960 1920 3600)  # s: then 2,4,8,16,32,60 min
API_RE='^(APIConnectionError|APIError|Timeout|InternalServerError'
API_RE+='|ServiceUnavailableError|BadGatewayError|APIResponseValidationError)$'
agent_s=0; eval_s=0

log() { echo "$(date '+%F %T') [$ID a$A] $*" | tee -a "$LOG" >> "$R/runs/driver.log"; }
memavail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
done_marker() {  # $1 = infra reason or empty
  local infra=""; [ -n "${1:-}" ] && infra=",\"infra\":\"$1\""
  echo "{\"id\":\"$ID\",\"attempt\":$A,\"agent_s\":$agent_s,\"eval_s\":$eval_s$infra}" \
    > "$REPORT"
}
cleanup_image() {
  podman rmi -f "$IMG" >/dev/null 2>&1
  podman image prune -f >/dev/null 2>&1
}
memkilled() { [ -f "$ST/memkill_$ID" ]; }
finish_memkill() {
  log "memory watchdog killed this instance; no done_ marker, will re-run"
  rm -f "$ST/memkill_$ID" "$ST/active/$ID"
  cleanup_image; exit 3
}

# --- memory guard: wait for headroom before starting
while [ "$(memavail_gb)" -lt 12 ]; do
  log "waiting: MemAvailable $(memavail_gb)G < 12G"; sleep 30
done
"$R/scripts/podman_service.sh" || { log "podman service down"; exit 4; }
echo "$IMG" > "$ST/active/$ID"; rm -f "$ST/memkill_$ID"

# --- a. pre-pull, globally rate limited, with backoff
pull_once() {
  (
    flock 9
    last=$(cat "$ST/last_pull" 2>/dev/null || echo 0)
    gap=$(( last + PULL_GAP - $(date +%s) ))
    [ "$gap" -gt 0 ] && sleep "$gap"
    date +%s > "$ST/last_pull"
    nice -n 19 ionice -c3 timeout 3600 podman pull -q "$IMG"
  ) 9>"$ST/pull.lock" >> "$LOG" 2>&1
}
pulled=0
for d in "${PULL_BACKOFF[@]}"; do
  if [ "$d" -gt 0 ]; then log "pull failed, retry in $((d / 60)) min"; sleep "$d"; fi
  if podman image exists "$IMG" || pull_once; then pulled=1; break; fi
  if tail -5 "$LOG" | grep -qiE 'manifest unknown|not found|name unknown'; then
    log "image does not exist upstream"; break
  fi
done
if [ $pulled = 0 ]; then
  log "pull_failed"; done_marker pull_failed; rm -f "$ST/active/$ID"; exit 0
fi

# --- b/c. agent; on API/connection failure re-run from scratch (max 3 tries)
wait_endpoint() {  # up to 30 min; the agent itself retries ~20 min per call
  for _ in $(seq 1 60); do
    curl -s -m 10 "$FLINT_API_BASE/models" | grep -q flint && return 0
    log "endpoint $FLINT_API_BASE down, waiting"; sleep 30
  done
  return 1
}
exit_status() {
  "$R/.venv/bin/python" -I - "$OUT/agent/$ID" "$ID" <<'PY'
import glob, sys, yaml
d, iid = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"{d}/exit_statuses_*.yaml"), reverse=True):
    data = yaml.safe_load(open(f)) or {}
    for st, ids in (data.get("instances_by_exit_status") or {}).items():
        if iid in (ids or []):
            print(st); sys.exit()
PY
}
cd "$R"
status=""
for try in 1 2 3; do
  if [ -d "$OUT/agent/$ID" ]; then
    mkdir -p "$OUT/agent_retired"
    mv "$OUT/agent/$ID" "$OUT/agent_retired/$ID.$(date +%s)"
  fi
  wait_endpoint
  log "agent start (try $try)"
  t0=$(date +%s)
  PYTHONPATH=$R/adapter timeout 6h nice -n 19 ionice -c3 \
    "$R/.venv/bin/mini-extra" swebench --subset verified --split test \
    --filter "^${ID}\$" -c swebench.yaml -c "$R/flint.yaml" \
    -c "model.model_kwargs.api_base=$FLINT_API_BASE" \
    -w 1 -o "$OUT/agent/$ID" > "$OUT/log/$ID.agent.log" 2>&1
  agent_s=$(( agent_s + $(date +%s) - t0 ))
  memkilled && finish_memkill
  status=$(exit_status)
  log "agent exit_status=${status:-?} (${agent_s}s total)"
  [[ "$status" =~ $API_RE ]] || break
  [ $try -lt 3 ] && log "API failure, re-running from scratch"
done
if [[ "$status" =~ $API_RE ]]; then
  log "api_failed"; done_marker api_failed
  rm -f "$ST/active/$ID"; cleanup_image; exit 0
fi

# --- d. official eval through the podman docker-compatible socket
t1=$(date +%s)
cd "$OUT/eval"
if [ -s "$OUT/agent/$ID/preds.json" ]; then
  nice -n 19 ionice -c3 "$R/.venv/bin/python" -m swebench.harness.run_evaluation \
    -d SWE-bench/SWE-bench_Verified -s test -i "$ID" \
    -p "$OUT/agent/$ID/preds.json" --max_workers 1 -id "att${A}_$ID" \
    --timeout 1800 > "$OUT/log/$ID.eval.log" 2>&1
fi
eval_s=$(( $(date +%s) - t1 ))
memkilled && finish_memkill
done_marker ""
log "done (agent ${agent_s}s, eval ${eval_s}s)"
rm -f "$ST/active/$ID"
cleanup_image
