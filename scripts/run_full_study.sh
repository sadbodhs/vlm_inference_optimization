#!/usr/bin/env bash
# The full study: both arms, with repeats, real documents, and enough accuracy
# samples to resolve the frontier. Strictly sequential -- two experiments against
# one server contaminate each other.
#
#   scripts/run_full_study.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# A multi-hour run that dies silently is worse than one that crashes: the log just
# stops, and "is it still going?" has no answer you can trust. Announce the failure,
# and write an explicit terminal marker either way so progress can be read from the
# log alone rather than inferred from a process listing.
trap 'echo "### STUDY FAILED at line $LINENO (exit $?)"' ERR
trap 'echo "### STUDY EXITED $?"' EXIT

OUT="${OUT:-results}"
DATA="${DATA:-data/docvqa}"
H=docker/run_harness.sh
REPEATS="${REPEATS:-3}"
E3_N="${E3_N:-500}"

run_arm_suite () {
  local ARM="$1" ID="$2"
  echo "################################################################"
  echo "## ARM: $ID"
  echo "################################################################"
  docker/run_server.sh stop >/dev/null 2>&1 || true
  WAIT_S=900 docker/run_server.sh start "$ARM"

  echo; echo "--- E1 roofline x$REPEATS ---"
  $H python3 experiments/e1_roofline.py --arm "$ARM" --out "$OUT" \
      --n 12 --max-tokens 128 --repeats "$REPEATS"

  echo; echo "--- E2 TTFT vs vision tokens x$REPEATS ---"
  $H python3 experiments/e2_ttft_vs_tokens.py --arm "$ARM" --out "$OUT" \
      --n 8 --sizes 224,448,672,896,1120,1344 --repeats "$REPEATS"

  echo; echo "--- E0 saturation on REAL documents ---"
  $H python3 experiments/e0_saturation.py --arm "$ARM" --out "$OUT" \
      --n 80 --rates 1,2,4,8,16 --max-tokens 64 --manifest "$DATA/manifest.jsonl"

  echo; echo "--- E3 token budget, n=$E3_N ---"
  $H python3 experiments/e3_token_budget.py --arm "$ARM" --out "$OUT" \
      --manifest "$DATA/manifest.jsonl" --scorer anls --limit "$E3_N" \
      --max-tokens 64 --budgets 200704,451584,802816,1605632,3211264
}

# Arms to run; default is both. Pass paths to re-run a subset without redoing
# hours of completed work.
if [ "$#" -gt 0 ]; then
  for A in "$@"; do run_arm_suite "$A" "$(basename "$A" .yaml)"; done
else
  run_arm_suite arms/B0_vllm_awq_clean.yaml B0_clean
  run_arm_suite arms/B_vllm_awq.yaml        B_defaults
fi

echo; echo "### ALL ARMS COMPLETE"
echo; echo "################ plots ################"
$H python3 experiments/plot.py $(ls "$OUT"/sweeps/*.json | sed 's|^|/work/|' | tr '\n' ' ')

echo; echo "################ accuracy CIs ################"
for ARM in B0_vllm_awq_clean B_vllm_awq; do
  RUNS=$(ls -d "$OUT"/e3-budget-$ARM-px* 2>/dev/null | tr '\n' ' ')
  [ -n "$RUNS" ] && $H python3 tools/accuracy_ci.py \
      --manifest "$DATA/manifest.jsonl" --runs $RUNS || true
done
