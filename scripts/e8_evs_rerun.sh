#!/usr/bin/env bash
# E8: rerun the 7B EVS-0.75 live point after the server died mid-run at 12 cameras
# (2026-10-03; vlm-server is --rm, so its log was lost). Same arm, same search, but
# the server log is streamed to a file for the whole run so a crash leaves a trace.
#   scripts/gpu_lock.sh run e8-evs-rerun -- scripts/e8_evs_rerun.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/e8; ARM="$OUT/arms/V_vllm_video-evs0.75.yaml"
grep -q -- "--video-pruning-rate 0.75" "$ARM" || { echo "missing $ARM" >&2; exit 2; }
docker/run_server.sh stop >/dev/null
GPU_UTIL=0.80 docker/run_server.sh start "$ARM" || exit 1
docker logs -f vlm-server > "$OUT/server-live-evs-rerun.log" 2>&1 &
LOGPID=$!
ALLOW_CONCURRENT=1 DS_TIMEOUT=5400 docker/run_ds.sh python3 experiments/e7c.py live \
  --arm-file arms/V_vllm_video.yaml --gate track-motion --arm full --input vid8 \
  --cams "${CAMS:-9,12}" --discard-first --out "$OUT/V_vllm_video/vid8-e75-rerun"
echo "server alive after run: $(docker ps -q -f name=vlm-server | wc -l)"
kill $LOGPID 2>/dev/null
docker/run_server.sh stop
echo "### EVS rerun COMPLETE $(date +%H:%M)"
