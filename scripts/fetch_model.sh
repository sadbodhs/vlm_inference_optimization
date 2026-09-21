#!/usr/bin/env bash
# Pre-fetch a model into the shared HF cache, so the first server start is not
# also a download. Runs in a container like everything else.
#
#   scripts/fetch_model.sh Qwen/Qwen2.5-VL-7B-Instruct-AWQ
#   scripts/fetch_model.sh arms/B_vllm_awq.yaml          # reads `model:` from the arm
set -euo pipefail

ARG="${1:?usage: $0 <model-id|arm.yaml>}"
if [ -f "$ARG" ]; then
  MODEL=$(grep -E '^model:' "$ARG" | head -1 | sed 's/^model:[[:space:]]*//')
else
  MODEL="$ARG"
fi
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

echo "fetching $MODEL -> $HF_CACHE"

# The model id travels as an env var, never inside nested quotes: this command
# crosses ssh -> bash -> docker -> bash -> python, and each layer eats a quoting
# level. An id interpolated into the python source arrives as a bare identifier.
docker run --rm \
  -v "$HF_CACHE:/hf" \
  -e HF_HOME=/hf \
  -e MODEL_ID="$MODEL" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  python:3.12-slim bash -c '
    set -euo pipefail
    pip install --quiet --disable-pip-version-check --root-user-action=ignore huggingface_hub
    # hf_transfer is retired; Xet high-performance mode is the current switch.
    export HF_XET_HIGH_PERFORMANCE=1
    python - <<PY
import os
from huggingface_hub import snapshot_download
path = snapshot_download(os.environ["MODEL_ID"])
print("OK", path)
PY
    # pip needs root inside the container, but the cache is shared with tools that
    # run as the host user (scripts/fetch_dataset.sh). Leaving root-owned blobs in
    # a shared cache breaks them later, so hand ownership back on the way out.
    chown -R "$HOST_UID:$HOST_GID" /hf
  '
echo "cache now: $(du -sh "$HF_CACHE" | cut -f1)"
