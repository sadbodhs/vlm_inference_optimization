#!/usr/bin/env bash
# Stage 2 of the comparison: the as-shipped arms, then the natural-image datasets.
#
# The clean arms (B0/C0) answer "which engine is faster with both vendors' default
# optimisations switched off". These two answer "what does each vendor's default
# configuration actually buy", which is a different and equally useful question --
# and the only honest way to read a vendor's own benchmark numbers.
set -euo pipefail
cd ~/sadbodh/vlm_inference_optimization
trap 'echo "### STAGE2 FAILED at line $LINENO"' ERR
trap 'echo "### STAGE2 EXITED $?"' EXIT
H=docker/run_harness.sh

for ARM in arms/B_vllm_awq.yaml arms/C_sglang_awq.yaml; do
  ID=$(basename "$ARM" .yaml)
  echo; echo "######## $ID (as shipped) ########"
  docker/run_server.sh stop >/dev/null 2>&1 || true
  WAIT_S=1200 docker/run_server.sh start "$ARM"

  $H python3 experiments/e1_roofline.py --arm "$ARM" --out results \
     --n 12 --max-tokens 128 --repeats 3
  $H python3 experiments/e2_ttft_vs_tokens.py --arm "$ARM" --out results \
     --n 8 --sizes 224,448,672,896,1120,1344 --repeats 3
  $H python3 experiments/e0_saturation.py --arm "$ARM" --out results --n 80 \
     --rates 0.5,1,1.5,2,4 --max-tokens 64 --manifest data/docvqa/manifest.jsonl --resize client
  $H python3 experiments/e3_token_budget.py --arm "$ARM" --out results \
     --manifest data/docvqa/manifest.jsonl --scorer anls --limit 500 --max-tokens 64 \
     --budgets 200704,451584,802816,1605632,3211264 --resize client
done

echo; echo "### STAGE2 COMPLETE"
