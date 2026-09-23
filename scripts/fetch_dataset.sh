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
  # No text at all: the pure-appearance arm the frontier claim actually needs.
  vqav2)   DS=lmms-lab/VQAv2;    CFG=default; SPLIT="${SPLIT:-validation}"; OUT=data/vqav2 ;;
  *) echo "unknown dataset" >&2; exit 2 ;;
esac

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE" "$OUT"

# Runs as root so pip can install, then hands the written files back to the host
# user on the way out. Running as the host uid instead leaves pip with no HOME and
# no writable site-packages, which fails at install time rather than at run time.
docker run --rm \
  -v "$HF_CACHE:/hf" -v "$PWD:/work" -w /work \
  -e HF_HOME=/hf \
  -e DS="$DS" -e CFG="$CFG" -e SPLIT="$SPLIT" -e N="$N" -e OUT="$OUT" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  python:3.12-slim bash -c '
    set -euo pipefail
    pip install --quiet --disable-pip-version-check --root-user-action=ignore datasets pillow
    python3 tools/build_manifest.py --dataset "$DS" --config "$CFG" \
        --split "$SPLIT" --n "$N" --out "$OUT"
    chown -R "$HOST_UID:$HOST_GID" "$OUT" /hf
  '

# Verify the thing we claim to have produced. Every silent failure in this repo so
# far has been a command that printed an error and still exited 0; a script that
# checks its own output cannot lie about having run.
MANIFEST="$OUT/manifest.jsonl"
if [ ! -s "$MANIFEST" ]; then
  echo "FAILED: $MANIFEST is missing or empty" >&2
  exit 1
fi
COUNT=$(wc -l < "$MANIFEST")
IMAGES=$(find "$OUT/images" -type f 2>/dev/null | wc -l)
echo "ok: $COUNT manifest rows, $IMAGES images in $OUT"
[ "$COUNT" -eq "$IMAGES" ] || { echo "FAILED: $COUNT rows but $IMAGES images" >&2; exit 1; }
