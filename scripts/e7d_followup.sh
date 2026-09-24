#!/usr/bin/env bash
# E7d follow-up, run after `PASS=live scripts/e7d.sh` finishes:
#   1. Qwen3-VL-8B again, both configurations, with one discarded warm-up run. Its
#      first live sweep failed 6 and 8 cameras on detection latency right after the
#      server started (1.8 s, 1.2 s p99), then passed 5; and its ROI sweep lost its
#      summary to a teardown hang. Reported as a rerun, alongside the first sweep.
#   2. PLAN.md 13's memory test: the <= 4B model with the most full-frame cameras at
#      util 0.40. If the rule picks a model whose answers were unusable in
#      recognition (a 2B), the best usable <= 4B model is run too, and both reported.
#   3. Release the shared GPU lock taken for E7d.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/e7d/live.log
until grep -q "### E7D live COMPLETE" "$LOG" 2>/dev/null; do sleep 60; done
echo "### follow-up start $(date +%H:%M)"

a=E_q3vl_8b; out=results/e7d/$a/rerun
mkdir -p "$out"
docker/run_server.sh stop >/dev/null
if GPU_UTIL=0.80 docker/run_server.sh start "arms/$a.yaml"; then
  for cfg in "track-motion full" "person-track-motion roi"; do
    set -- $cfg
    echo "--- $a rerun  $1  $2"
    ALLOW_CONCURRENT=1 DS_TIMEOUT=5400 docker/run_ds.sh python3 experiments/e7c.py live \
      --arm-file "arms/$a.yaml" --gate "$1" --arm "$2" --cams 6,8,10,12,14,16,18,20,22,24 \
      --search --discard-first --out "$out"
    sleep 10
  done
fi
docker logs vlm-server > "$out/server.log" 2>&1 || true
docker/run_server.sh stop >/dev/null

# the memory test: pick by the pre-registered rule
PICKS=$(python3 - <<'EOF'
import json, pathlib
small = ["E_q25_3b", "E_q35_4b", "E_q3vl_4b", "E_q3vl_2b", "E_q35_2b"]
unusable = {"E_q3vl_2b", "E_q35_2b"}     # recognition: ticks every box / never answers in format
best = {}
for a in small:
    f = pathlib.Path(f"results/e7d/{a}/track-motion-full-deepstream.json")
    if f.exists():
        ok = [r["cams"] for r in json.loads(f.read_text())["rows"] if r["supported"]]
        best[a] = max(ok) if ok else 0
rule = max(best, key=best.get)
usable = max((a for a in best if a not in unusable), key=best.get)
print(rule if rule == usable else f"{rule} {usable}")
EOF
)
echo "### memory test at util 0.40 for: $PICKS"
PASS=mem UTIL=0.40 ARMS="$PICKS" scripts/e7d.sh

scripts/gpu_lock.sh release vlm-e7d
echo "### E7D FOLLOW-UP COMPLETE $(date +%H:%M)"
