#!/usr/bin/env bash
# E9: rerun live points that ran against a dead server (the sweep's next count
# after a crash). Fresh server, one discarded warm-up, server log kept.
#   RERUN="pre-0.5 vlmbench-vllm-pevs:v0.29.0 0.5 vid8 9" scripts/gpu_lock.sh run e9-rerun -- scripts/e9_rerun.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/e9; T0=$(date +%s)
while read -r label image rate input cams; do
  [ -n "$label" ] || continue
  arm="$OUT/arms/V_vllm_video-rate$rate.yaml"
  sed -E "s/^(server_args:.*)$/\1 --video-pruning-rate $rate/" arms/V_vllm_video.yaml > "$arm"
  echo "--- rerun $label at $cams cameras $(date +%H:%M)"
  docker/run_server.sh stop >/dev/null < /dev/null
  VLLM_IMAGE=$image GPU_UTIL=0.80 docker/run_server.sh start "$arm" < /dev/null || { echo "--- server FAILED"; continue; }
  docker logs -f vlm-server > "$OUT/server-rerun-$label.log" 2>&1 &
  ALLOW_CONCURRENT=1 DS_TIMEOUT=5400 docker/run_ds.sh python3 experiments/e7c.py live \
    --arm-file arms/V_vllm_video.yaml --gate track-motion --arm full --input "$input" \
    --cams "$cams" --discard-first --out "$OUT/live/$label-rerun" < /dev/null
  echo "  server alive after rerun: $(docker ps -q -f name=vlm-server | wc -l)"
done <<< "$RERUN"
docker/run_server.sh stop < /dev/null
printf 'D-rerun\t%s\t%s\t%.2f\t%s\n' "$(date -d @$T0 +%FT%T)" "$(date +%FT%T)" \
  "$(echo "($(date +%s)-$T0)/3600" | bc -l)" "$(echo "$RERUN" | awk '{print $1"@"$5}' | paste -sd,)" >> "$OUT/gpu_ledger.tsv"
echo "### E9 rerun COMPLETE $(date +%H:%M)"
