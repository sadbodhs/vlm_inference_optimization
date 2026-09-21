#!/usr/bin/env bash
# Full dry run, entirely in Docker. No GPU, no host Python, no venv.
# Validates arm configs, the harness and the plotting path before the 3090 is
# occupied. Build once:  docker build -f docker/harness.Dockerfile -t vlmbench-harness .
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results/dryrun}"
ARM=arms/Z_mock.yaml
H=docker/run_harness.sh

docker image inspect vlmbench-harness:latest >/dev/null 2>&1 || {
  echo "building harness image..."
  docker build -q -f docker/harness.Dockerfile -t vlmbench-harness .
}

docker/run_mock.sh start
trap 'docker/run_mock.sh stop >/dev/null 2>&1 || true' EXIT
sleep 2

COMMON=(--arm "$ARM" --out "$OUT" --dry-run)

echo "=== E1 roofline ==="
$H python3 experiments/e1_roofline.py "${COMMON[@]}" --n 8 --max-tokens 48

echo; echo "=== E2 TTFT vs vision tokens ==="
$H python3 experiments/e2_ttft_vs_tokens.py "${COMMON[@]}" --n 5 --sizes 224,448,672,896,1120

echo; echo "=== E0 saturation ==="
$H python3 experiments/e0_saturation.py "${COMMON[@]}" --n 40 --rates 2,4,8,16,24 --max-tokens 32

echo; echo "=== E3 token budget (with accuracy, demo set) ==="
$H python3 experiments/e3_token_budget.py "${COMMON[@]}" --limit 4 --max-tokens 12 \
  --manifest data/demo/manifest.jsonl --scorer anls --budgets 200704,802816,1605632

echo; echo "=== plots ==="
$H python3 experiments/plot.py $(ls "$OUT"/sweeps/*.json | sed "s|^|/work/|" | tr '\n' ' ')
