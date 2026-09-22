#!/usr/bin/env bash
# Generational comparison: run the identical experiment set on each model.
#
#   scripts/run_model_comparison.sh arms/Q3_vllm_awq_clean.yaml
#
# Same stack, same harness image, same prompts, same client-side token budget,
# same datasets. Only the checkpoint changes, so a difference in the table is the
# model rather than the serving configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'echo "### MODELCMP FAILED at line $LINENO"' ERR
trap 'echo "### MODELCMP EXITED $?"' EXIT

H=docker/run_harness.sh
OUT="${OUT:-results}"
DATA="${DATA:-data/docvqa}"
REPEATS="${REPEATS:-3}"
E3_N="${E3_N:-500}"
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(arms/Q3_vllm_awq_clean.yaml)

for ARM in "${ARMS[@]}"; do
  ID=$(basename "$ARM" .yaml)
  echo; echo "######## $ID ########"
  docker/run_server.sh stop >/dev/null 2>&1 || true
  WAIT_S=1200 docker/run_server.sh start "$ARM"

  # The roofline basis must come from the server, not from an estimate. Print it
  # so a wrong weight_bytes in the arm is visible before E1 uses it.
  echo "--- server's own memory report ---"
  docker logs vlm-server 2>&1 | grep -iE "Model loading took|KV cache size" | tail -2

  $H python3 experiments/e1_roofline.py --arm "$ARM" --out "$OUT" \
     --n 12 --max-tokens 128 --repeats "$REPEATS"
  $H python3 experiments/e2_ttft_vs_tokens.py --arm "$ARM" --out "$OUT" \
     --n 8 --sizes 224,448,672,896,1120,1344 --repeats "$REPEATS"
  $H python3 experiments/e0_saturation.py --arm "$ARM" --out "$OUT" --n 80 \
     --rates 0.5,1,1.5,2,4 --max-tokens 64 --manifest "$DATA/manifest.jsonl" --resize client
  $H python3 experiments/e3_token_budget.py --arm "$ARM" --out "$OUT" \
     --manifest "$DATA/manifest.jsonl" --scorer anls --limit "$E3_N" --max-tokens 64 \
     --budgets 200704,451584,802816,1605632,3211264 --resize client
done

echo; echo "### MODELCMP COMPLETE"
