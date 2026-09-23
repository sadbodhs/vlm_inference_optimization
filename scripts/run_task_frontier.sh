#!/usr/bin/env bash
# The per-task frontier: the same token-budget sweep on datasets that need
# different amounts of resolution.
#
# DocVQA is dense text on a page. ChartQA is structured but sparser. TextVQA is a
# natural scene with incidental text. If the saturation point moves across them,
# "how many vision tokens do I need" has no single answer -- which is the claim
# the DocVQA-only result could not support.
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'echo "### FRONTIER FAILED at line $LINENO"' ERR
trap 'echo "### FRONTIER EXITED $?"' EXIT

ARM="${ARM:-arms/B0_vllm_awq_clean.yaml}"
H=docker/run_harness.sh
LIMIT="${LIMIT:-400}"

docker/run_server.sh stop >/dev/null 2>&1 || true
WAIT_S=1200 docker/run_server.sh start "$ARM"

# scorer per dataset: the official metric, not a convenient one
run () {
  local data="$1" scorer="$2" tag="$3"
  [ -s "$data/manifest.jsonl" ] || { echo "  skip $tag: no manifest"; return 0; }
  echo; echo "######## $tag ($scorer) ########"
  $H python3 experiments/e3_token_budget.py --arm "$ARM" --out results \
     --manifest "$data/manifest.jsonl" --scorer "$scorer" --limit "$LIMIT" \
     --max-tokens 64 --budgets 50176,200704,451584,802816,1605632 \
     --resize client --tag "e3-$tag"
}

run data/docvqa  anls    docvqa
run data/chartqa relaxed chartqa
run data/textvqa em      textvqa

echo; echo "### FRONTIER COMPLETE"
