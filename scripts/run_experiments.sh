#!/usr/bin/env bash
# Run the full experiment set against a real server, in dependency order.
#
#   scripts/run_experiments.sh arms/B_vllm_awq.yaml
#
# Order is not arbitrary:
#   E1 first  -- it validates the harness. A decode rate above the memory-bandwidth
#                roofline is physically impossible, so if E1 fails that check,
#                nothing measured afterwards can be trusted and the run stops.
#   E2 second -- concurrency 1, establishes the TTFT vs vision-token curve with no
#                queueing in the way.
#   E0 third  -- finds the saturation point. Needed before any throughput claim.
#   E3 last   -- the expensive one, and the only one that needs a dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${1:?usage: $0 <arm.yaml> [dataset-dir]}"
DATA="${2:-data/docvqa}"
OUT="${OUT:-results}"
ID=$(grep -E '^id:' "$ARM" | head -1 | sed 's/^id:[[:space:]]*//')
H=docker/run_harness.sh

command -v curl >/dev/null && SERVER_UP=$(curl -sf "http://127.0.0.1:${VLLM_PORT:-8000}/v1/models" >/dev/null 2>&1 && echo yes || echo no)
[ "${SERVER_UP:-no}" = "yes" ] || { echo "no server on :${VLLM_PORT:-8000} -- start it with docker/run_vllm.sh start $ARM" >&2; exit 1; }

echo "########## E1 roofline (harness validation) ##########"
$H python3 experiments/e1_roofline.py --arm "$ARM" --out "$OUT" --n 12 --max-tokens 128

ROOF_JSON="$OUT/sweeps/e1-roofline-$ID.json"
$H python3 - <<PY
import json, sys
d = json.load(open("$ROOF_JSON"))
r = d["rows"][0]
pct = r.get("pct_of_roofline")
print(f"  measured {r['decode_tok_s']:.1f} tok/s = {pct:.1f}% of the {r['roofline_tok_s']:.0f} tok/s roofline")
if pct is None:
    sys.exit("E1 produced no rate; aborting")
if pct > 100:
    sys.exit("E1 ABOVE ROOFLINE -- harness is wrong. Refusing to run the rest.")
if pct < 15:
    print("  NOTE: far below roofline. Expected for a first run (no CUDA graphs,")
    print("  cold caches) but worth explaining before publishing.")
PY

echo; echo "########## E2 TTFT vs vision tokens ##########"
$H python3 experiments/e2_ttft_vs_tokens.py --arm "$ARM" --out "$OUT" \
    --n 8 --sizes 224,448,672,896,1120,1344

echo; echo "########## E0 saturation ##########"
$H python3 experiments/e0_saturation.py --arm "$ARM" --out "$OUT" \
    --n 60 --rates 0.5,1,2,4,8 --max-tokens 64

echo; echo "########## E3 token budget x accuracy ##########"
if [ -s "$DATA/manifest.jsonl" ]; then
  $H python3 experiments/e3_token_budget.py --arm "$ARM" --out "$OUT" \
      --manifest "$DATA/manifest.jsonl" --scorer anls --limit 100 --max-tokens 64 \
      --budgets 200704,451584,802816,1605632,3211264
else
  echo "  skipped: $DATA/manifest.jsonl missing (scripts/fetch_dataset.sh docvqa)"
fi

echo; echo "########## plots ##########"
$H python3 experiments/plot.py $(ls "$OUT"/sweeps/*.json | sed 's|^|/work/|' | tr '\n' ' ')
