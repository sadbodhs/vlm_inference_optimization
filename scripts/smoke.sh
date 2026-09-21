#!/usr/bin/env bash
# Full dry run: every experiment against the mock server, no GPU required.
# Use this to validate an arm config and the plotting path before occupying the 3090.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8077}"
OUT="${OUT:-results/dryrun}"

python3 tools/mock_vlm_server.py --port "$PORT" --max-running 8 --itl-ms 12 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT
sleep 1.5

ARM=arms/Z_mock.yaml
COMMON=(--arm "$ARM" --out "$OUT" --dry-run --mock-url "http://127.0.0.1:$PORT")

echo "=== E1 roofline ==="
python3 experiments/e1_roofline.py "${COMMON[@]}" --n 8 --max-tokens 48

echo; echo "=== E2 TTFT vs vision tokens ==="
python3 experiments/e2_ttft_vs_tokens.py "${COMMON[@]}" --n 5 --sizes 224,448,672,896,1120

echo; echo "=== E0 saturation ==="
python3 experiments/e0_saturation.py "${COMMON[@]}" --n 40 --rates 2,4,8,16,24 --max-tokens 32

echo; echo "=== E3 token budget (latency axis only) ==="
python3 experiments/e3_token_budget.py "${COMMON[@]}" --limit 6 --max-tokens 16 \
  --budgets 200704,451584,802816,1605632

echo; echo "=== plots ==="
python3 experiments/plot.py "$OUT"/sweeps/*.json
