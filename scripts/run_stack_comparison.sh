#!/usr/bin/env bash
# Thread 1: the same model on different serving stacks.
#
#   scripts/run_stack_comparison.sh
#
# Runs the identical experiment set against each arm in turn, one stack at a time.
# Nothing about the measurement changes between arms -- same harness container,
# same prompts, same images, same scorer -- so a difference in the table is a
# difference in the stack (R1).
#
# Pairing is deliberate: the two "clean" arms have each vendor's default-on
# optimisations switched OFF, because vendors enable different amounts of work by
# default and comparing defaults would attribute the difference to the wrong thing.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results}"
DATA="${DATA:-data/docvqa}"
REPEATS="${REPEATS:-3}"
E3_N="${E3_N:-300}"
H=docker/run_harness.sh
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(
  arms/B0_vllm_awq_clean.yaml
  arms/C0_sglang_awq_clean.yaml
  arms/B_vllm_awq.yaml
  arms/C_sglang_awq.yaml
)

trap 'echo "### STACK COMPARISON FAILED at line $LINENO"' ERR
trap 'echo "### STACK COMPARISON EXITED $?"' EXIT

for ARM in "${ARMS[@]}"; do
  ID=$(basename "$ARM" .yaml)
  echo; echo "################################################################"
  echo "## $ID"
  echo "################################################################"
  docker/run_server.sh stop >/dev/null 2>&1 || true
  WAIT_S=1200 docker/run_server.sh start "$ARM"

  echo; echo "--- E1 decode x$REPEATS ---"
  $H python3 experiments/e1_roofline.py --arm "$ARM" --out "$OUT" \
      --n 12 --max-tokens 128 --repeats "$REPEATS"

  echo; echo "--- E2 TTFT vs vision tokens x$REPEATS ---"
  $H python3 experiments/e2_ttft_vs_tokens.py --arm "$ARM" --out "$OUT" \
      --n 8 --sizes 224,448,672,896,1120,1344 --repeats "$REPEATS"

  echo; echo "--- E0 saturation, real documents, zero reuse ---"
  $H python3 experiments/e0_saturation.py --arm "$ARM" --out "$OUT" \
      --n 80 --rates 0.5,1,1.5,2,4 --max-tokens 64 \
      --manifest "$DATA/manifest.jsonl"

  echo; echo "--- E3 token budget, n=$E3_N ---"
  $H python3 experiments/e3_token_budget.py --arm "$ARM" --out "$OUT" \
      --manifest "$DATA/manifest.jsonl" --scorer anls --limit "$E3_N" \
      --max-tokens 64 --budgets 200704,451584,802816,1605632
done

echo; echo "### STACK COMPARISON COMPLETE"
