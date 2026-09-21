#!/usr/bin/env bash
# Freeze an eval subset into data/<name>/. Containerised like everything else.
#   scripts/fetch_dataset.sh docvqa
#   scripts/fetch_dataset.sh chartqa
#   scripts/fetch_dataset.sh textvqa
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-200}"
case "${1:?usage: $0 <docvqa|chartqa|textvqa>}" in
  docvqa)  DS=lmms-lab/DocVQA;  CFG=DocVQA;  SPLIT="${SPLIT:-validation}"; OUT=data/docvqa ;;
  chartqa) DS=lmms-lab/ChartQA; CFG=default; SPLIT="${SPLIT:-test}";       OUT=data/chartqa ;;
  textvqa) DS=lmms-lab/textvqa; CFG=default; SPLIT="${SPLIT:-validation}"; OUT=data/textvqa ;;
  *) echo "unknown dataset" >&2; exit 2 ;;
esac

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE" "$OUT"

docker run --rm \
  -v "$HF_CACHE:/hf" -v "$PWD:/work" -w /work \
  -e HF_HOME=/hf -u "$(id -u):$(id -g)" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  python:3.12-slim bash -c "
    set -euo pipefail
    pip install --quiet --disable-pip-version-check datasets pillow
    python3 tools/build_manifest.py --dataset '$DS' --config '$CFG' \
        --split '$SPLIT' --n '$N' --out '$OUT'
  "
