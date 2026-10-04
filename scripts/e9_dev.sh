#!/usr/bin/env bash
# E9 development checks for the pre-encoder pruning plugin (vlmbench-vllm-pevs).
# Each spec: name rate mode check clip-every limit-clips
#   diag-*  PEVS_CHECK=1: kept-token drift vs the full encoder, encoder time, and the
#           keep-all identity test (development only: doubles encoder work)
#   acc-*   answers on one clip per duty-cycle bin (4 clips, 600 windows)
#   scripts/gpu_lock.sh run e9-dev -- scripts/e9_dev.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export VLLM_IMAGE=vlmbench-vllm-pevs:v0.29.0
OUT=results/e9/dev; mkdir -p "$OUT/arms"
SPECS=${SPECS:-"diag-window 0.675 window 1 1 1
acc-unit 0.675 unit 0 6 4
acc-window 0.675 window 0 6 4"}
T0=$(date +%s)
while read -r name rate mode check every lim; do
  [ -n "$name" ] || continue
  arm="$OUT/arms/pevs-$rate.yaml"
  sed -E "s/^(server_args:.*)$/\1 --video-pruning-rate $rate/" arms/V_vllm_video.yaml > "$arm"
  echo "### $name rate $rate mode $mode $(date +%H:%M)"
  docker/run_server.sh stop >/dev/null < /dev/null
  if ! PEVS_MODE=$mode PEVS_CHECK=$check GPU_UTIL=0.80 docker/run_server.sh start "$arm"; then
    echo "### server failed"; continue
  fi
  docker logs -f vlm-server > "$OUT/server-$name.log" 2>&1 &
  docker/run_harness.sh python3 experiments/e8_vlm.py --arm arms/V_vllm_video.yaml \
    --inputs vid8 --evs "$rate" --clip-every "$every" --limit-clips "$lim" --concurrency 8 --out "$OUT/$name" < /dev/null
  sleep 2
  grep -E "vlm_prune|PEVS_CHECK" "$OUT/server-$name.log" | sed "s/.*\] //" | tail -4
done <<< "$SPECS"
docker/run_server.sh stop
printf 'C-dev\t%s\t%s\t%.2f\t%s\n' "$(date -d @$T0 +%FT%T)" "$(date +%FT%T)" \
  "$(echo "($(date +%s)-$T0)/3600" | bc -l)" "$(echo "$SPECS" | awk '{print $1}' | paste -sd,)" >> results/e9/gpu_ledger.tsv
echo "### E9 dev COMPLETE $(date +%H:%M)"
