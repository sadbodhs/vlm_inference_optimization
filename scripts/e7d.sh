#!/usr/bin/env bash
# E7d (PLAN.md 13): which VLM behind the gate -- size and generation.
#
#   PASS=recog scripts/e7d.sh        # recognition, every model (E7's VLM stage)
#   PASS=live  scripts/e7d.sh        # cameras per 3090, every model (E7c's live run)
#   PASS=mem   ARMS=E_q35_2b scripts/e7d.sh   # the chosen model again at util 0.40
#
# One model at a time on the card; a model that fails is logged and skipped, so one
# bad checkpoint does not cost the night. Server logs are kept per model: vLLM's
# allocator retries (E7b) only show up there.
set -uo pipefail
cd "$(dirname "$0")/.."
PASS="${PASS:-recog}"
ARMS="${ARMS:-E_q25_3b E_q35_4b E_q35_2b E_q3vl_4b E_q3vl_2b E_q3vl_8b E_q35_9b}"
UTIL="${UTIL:-0.80}"
CAMS="${CAMS:-6,8,10,12,14,16,18,20,22,24}"

smoke () {   # one real two-frame request: the model answers, and in letters
  python3 - "$1" <<'EOF'
import base64, json, sys, urllib.request, yaml
arm = yaml.safe_load(open(f"arms/{sys.argv[1]}.yaml"))
jpg = base64.b64encode(open(sorted(__import__("glob").glob("data/meva/frames/*/*.jpg"))[0], "rb").read()).decode()
img = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg}"}}
body = {"model": arm["model"], "max_tokens": 16, "temperature": 0,
        "messages": [{"role": "user", "content": [img, img, {"type": "text", "text":
            "Which of these is happening? A. A person walks. B. A car turns. "
            "Answer with letters separated by commas, or N if none apply."}]}],
        # the experiments cap each frame at 451,584 px client-side; so does the check,
        # or two raw 1080p frames (~4,100 tokens) overflow a 4,096-token context
        "mm_processor_kwargs": {"max_pixels": 451584},
        **(arm.get("extra_body") or {})}
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
    {"Content-Type": "application/json"}), timeout=300))
txt = r["choices"][0]["message"]["content"]
print(f"  smoke: {txt!r}  prompt tokens {r['usage']['prompt_tokens']}")
sys.exit(1 if "<think>" in (txt or "") or not (txt or "").strip() else 0)
EOF
}

one () {
  local a="$1" out="results/e7d/$1"
  mkdir -p "$out"
  docker/run_server.sh stop >/dev/null
  GPU_UTIL="$UTIL" docker/run_server.sh start "arms/$a.yaml" || return 1
  smoke "$a" || return 1
  case "$PASS" in
    recog)
      docker/run_harness.sh python3 experiments/e7_vlm.py --arm "arms/$a.yaml" --arms full,roi || return 1
      docker/run_harness.sh python3 experiments/e7_report.py --arm-id "$a" \
        --out "$out/cascades.json" > "$out/report.txt" || return 1
      tail -12 "$out/report.txt"
      ;;
    live|mem)
      local sub="$out"; [ "$PASS" = mem ] && sub="$out/util$UTIL"
      for cfg in "track-motion full" "person-track-motion roi"; do
        set -- $cfg
        echo "--- $a  $1  $2  (util $UTIL)"
        ALLOW_CONCURRENT=1 DS_TIMEOUT=5400 docker/run_ds.sh python3 experiments/e7c.py live \
          --arm-file "arms/$a.yaml" --gate "$1" --arm "$2" --cams "$CAMS" --search \
          --out "$sub" || return 1
        sleep 10
      done
      ;;
  esac
}

for a in $ARMS; do
  echo "### $a  pass=$PASS  $(date +%H:%M)"
  if one "$a"; then echo "### $a DONE $(date +%H:%M)"; else echo "### $a FAILED $(date +%H:%M)"; fi
  docker logs vlm-server > "results/e7d/$a/server-$PASS.log" 2>&1 || true
  echo "  allocator OOM retries: $(grep -ciE 'out of memory|OutOfMemory|failed to allocate' "results/e7d/$a/server-$PASS.log" || true)"
done
docker/run_server.sh stop
echo "### E7D $PASS COMPLETE"
