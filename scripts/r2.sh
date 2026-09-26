#!/usr/bin/env bash
# R2 (RECIPE.md): hand crops on HA4M. Under the GPU lock:
#   scripts/gpu_lock.sh run r2 -- scripts/r2.sh
#   SMOKE=1 scripts/gpu_lock.sh run r2-smoke -- scripts/r2.sh     # 10 samples
# When R2 is published, delete data/ha4m (the user asked; RECIPE.md says so).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/r2; LIMIT=0
if [ "${SMOKE:-0}" = 1 ]; then OUT=results/r2-smoke; LIMIT=10; fi
mkdir -p "$OUT"

run_model () {  # arm arms
  docker/run_server.sh stop >/dev/null
  GPU_UTIL=0.80 docker/run_server.sh start "arms/$1.yaml" || return 1
  docker/run_harness.sh python3 experiments/r2_run.py --arm "arms/$1.yaml" --arms "$2" \
      --limit "$LIMIT" --r2-out "$OUT"
  local rc=$?
  docker logs vlm-server > "$OUT/server-$1.log" 2>&1 || true
  return $rc
}

echo "### R2 start $(date +%H:%M)  out=$OUT"
run_model E_q3vl_8b F,B,HK,HY,HKC,F-n,HK-n && echo "### E_q3vl_8b DONE $(date +%H:%M)" || echo "### E_q3vl_8b FAILED $(date +%H:%M)"
run_model E_q3vl_4b F,HK,HKC && echo "### E_q3vl_4b DONE $(date +%H:%M)" || echo "### E_q3vl_4b FAILED $(date +%H:%M)"
docker/run_server.sh stop
echo "### R2 COMPLETE $(date +%H:%M)"
