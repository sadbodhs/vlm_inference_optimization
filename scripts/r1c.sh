#!/usr/bin/env bash
# R1c (RECIPE.md): precision pass on R1b's gate-fired windows. Needs results/r1b
# (first-pass answers + windows.json). Under the GPU lock:
#   scripts/gpu_lock.sh run r1c -- scripts/r1c.sh
#   SMOKE=1 scripts/gpu_lock.sh run r1c-smoke -- scripts/r1c.sh     # 12 windows
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/r1c; LIMIT=0
if [ "${SMOKE:-0}" = 1 ]; then OUT=results/r1c-smoke; LIMIT=12; fi
mkdir -p "$OUT"

run_model () {  # arm arms
  docker/run_server.sh stop >/dev/null
  GPU_UTIL=0.80 docker/run_server.sh start "arms/$1.yaml" || return 1
  docker/run_harness.sh python3 experiments/r1c_run.py --arm "arms/$1.yaml" --arms "$2" \
      --limit "$LIMIT" --r1c-out "$OUT"
  local rc=$?
  docker logs vlm-server > "$OUT/server-$1.log" 2>&1 || true
  return $rc
}

echo "### R1c start $(date +%H:%M)  out=$OUT"
run_model E_q3vl_4b Fc,GOc,FV,GOV && echo "### E_q3vl_4b DONE $(date +%H:%M)" || echo "### E_q3vl_4b FAILED $(date +%H:%M)"
run_model E_q3vl_8b Fc,FV,F4V8 && echo "### E_q3vl_8b DONE $(date +%H:%M)" || echo "### E_q3vl_8b FAILED $(date +%H:%M)"
docker/run_server.sh stop
echo "### R1c COMPLETE $(date +%H:%M)"
