#!/usr/bin/env bash
# R1b (RECIPE.md): label-free group crops on gate-fired windows. Under the GPU lock:
#   scripts/gpu_lock.sh run r1b -- scripts/r1b.sh
#   SMOKE=1 scripts/gpu_lock.sh run r1b-smoke -- scripts/r1b.sh     # 6 windows
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/r1b; LIMIT=0
if [ "${SMOKE:-0}" = 1 ]; then OUT=results/r1b-smoke; LIMIT=6; fi
mkdir -p "$OUT"
# the windows and groups are recomputed here (deterministic, no GPU) so the rig
# does not depend on a results file synced from elsewhere
docker/run_harness.sh python3 experiments/r1b_windows.py --out "$OUT/windows.json" | tail -4

run_model () {  # arm arms
  docker/run_server.sh stop >/dev/null
  GPU_UTIL=0.80 docker/run_server.sh start "arms/$1.yaml" || return 1
  docker/run_harness.sh python3 experiments/r1b_run.py --arm "arms/$1.yaml" --arms "$2" \
      --windows "$OUT/windows.json" --limit "$LIMIT" --r1b-out "$OUT"
  local rc=$?
  docker logs vlm-server > "$OUT/server-$1.log" 2>&1 || true
  return $rc
}

echo "### R1b start $(date +%H:%M)  out=$OUT"
run_model E_q3vl_4b F,M,GS,GSC,GO && echo "### E_q3vl_4b DONE $(date +%H:%M)" || echo "### E_q3vl_4b FAILED $(date +%H:%M)"
run_model E_q3vl_8b F,GO && echo "### E_q3vl_8b DONE $(date +%H:%M)" || echo "### E_q3vl_8b FAILED $(date +%H:%M)"
docker/run_server.sh stop
echo "### R1b COMPLETE $(date +%H:%M)"
