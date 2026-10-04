#!/usr/bin/env bash
# E9 (PLAN.md 15): prune video patches before the vision encoder.
#   PASS=recog scripts/gpu_lock.sh run e9-recog -- scripts/e9.sh
#   PASS=live  scripts/gpu_lock.sh run e9-live  -- scripts/e9.sh
# Plugin arms run on vlmbench-vllm-pevs:v0.29.0 (v0.29.0 + plugins/vlm_prune);
# evs-0.5 on the stock image. Every pass appends its GPU time to results/e9/gpu_ledger.tsv.
set -uo pipefail
cd "$(dirname "$0")/.."
PASS="${PASS:-recog}"
CAMS="${CAMS:-6,8,10,12,14,16,18,20,22,24}"
OUT=results/e9; mkdir -p "$OUT/arms" "$OUT/live"
PEVS=vlmbench-vllm-pevs:v0.29.0
T0=$(date +%s)

serve () {  # image rate
  local arm="$OUT/arms/V_vllm_video-rate$2.yaml"
  sed -E "s/^(server_args:.*)$/\1 --video-pruning-rate $2/" arms/V_vllm_video.yaml > "$arm"
  docker/run_server.sh stop >/dev/null < /dev/null
  VLLM_IMAGE=$1 GPU_UTIL=0.80 docker/run_server.sh start "$arm" < /dev/null || return 1
  docker logs -f vlm-server > "$OUT/server-$PASS-$3.log" 2>&1 &
}

case "$PASS" in
  recog)
    while read -r label rate input; do
      echo "### recog $label $(date +%H:%M)"
      if serve "$PEVS" "$rate" "$label"; then
        docker/run_harness.sh python3 experiments/e8_vlm.py --arm arms/V_vllm_video.yaml \
          --inputs "$input" --evs "$rate" --out "$OUT" < /dev/null \
          && echo "### $label DONE $(date +%H:%M)" || echo "### $label FAILED $(date +%H:%M)"
      else echo "### $label server FAILED"; fi
    done <<'EOS'
pre-0.675 0.675 vid8
pre-0.5 0.5 vid8
pre-trk-0.675 0.675 vid8t
EOS
    ;;
  live)
    while read -r label image rate input; do
      echo "--- live $label $(date +%H:%M)"
      if serve "$image" "$rate" "$label"; then
        ALLOW_CONCURRENT=1 DS_TIMEOUT=5400 docker/run_ds.sh python3 experiments/e7c.py live \
          --arm-file arms/V_vllm_video.yaml --gate track-motion --arm full --input "$input" \
          --cams "$CAMS" --search --discard-first --out "$OUT/live/$label" < /dev/null \
          && echo "--- $label DONE $(date +%H:%M)" || echo "--- $label FAILED $(date +%H:%M)"
        echo "  server alive after sweep: $(docker ps -q -f name=vlm-server | wc -l)"
      else echo "--- $label server FAILED"; fi
      sleep 10
    done <<EOS
pre-0.675 $PEVS 0.675 vid8
pre-0.5 $PEVS 0.5 vid8
pre-trk-0.675 $PEVS 0.675 vid8t
evs-0.5 vllm/vllm-openai:v0.29.0 0.5 vid8
EOS
    ;;
esac
docker/run_server.sh stop < /dev/null
printf '%s\t%s\t%s\t%.2f\tscripts/e9.sh\n' "D-$PASS" "$(date -d @$T0 +%FT%T)" "$(date +%FT%T)" \
  "$(echo "($(date +%s)-$T0)/3600" | bc -l)" >> "$OUT/gpu_ledger.tsv"
echo "### E9 $PASS COMPLETE $(date +%H:%M)"
