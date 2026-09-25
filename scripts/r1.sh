#!/usr/bin/env bash
# R1 (RECIPE.md): actor-crop reference level. Run under the shared GPU lock:
#   scripts/gpu_lock.sh run r1 -- scripts/r1.sh            # full run
#   SMOKE=1 scripts/gpu_lock.sh run r1-smoke -- scripts/r1.sh   # 5 samples per kind
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/r1; LIMIT=0
if [ "${SMOKE:-0}" = 1 ]; then OUT=results/r1-smoke; LIMIT=5; fi
mkdir -p "$OUT"

run_model () {  # arm arms
  docker/run_server.sh stop >/dev/null
  GPU_UTIL=0.80 docker/run_server.sh start "arms/$1.yaml" || return 1
  docker/run_harness.sh python3 experiments/r1_actor_crops.py --arm "arms/$1.yaml" \
      --arms "$2" --limit "$LIMIT" --r1-out "$OUT"
  local rc=$?
  docker logs vlm-server > "$OUT/server-$1.log" 2>&1 || true
  return $rc
}

echo "### R1 start $(date +%H:%M)  out=$OUT"
run_model E_q3vl_4b A0,A1,A2-m1.2,A2-m1.5,A2-m2,A2-m3,A3-h112,A3-h224,A3-h448,A4,A5 \
  && echo "### E_q3vl_4b DONE $(date +%H:%M)" || echo "### E_q3vl_4b FAILED $(date +%H:%M)"
run_model E_q3vl_8b A0,A1,A2-m2,A4 \
  && echo "### E_q3vl_8b DONE $(date +%H:%M)" || echo "### E_q3vl_8b FAILED $(date +%H:%M)"
docker/run_server.sh stop
echo "### R1 COMPLETE $(date +%H:%M)"
