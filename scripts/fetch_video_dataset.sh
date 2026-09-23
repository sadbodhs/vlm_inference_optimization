#!/usr/bin/env bash
# Freeze a video benchmark into data/<name>/. Containerised like everything else.
#   scripts/fetch_video_dataset.sh tempcompass
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-300}"
case "${1:?usage: $0 <tempcompass|nextqa>}" in
  tempcompass) DS=lmms-lab/TempCompass; CFG=multi-choice; SPLIT=test
               ARCHIVE=tempcompass_videos.zip; OUT=data/tempcompass ;;
  nextqa)      DS=lmms-lab/NExTQA; CFG=MC; SPLIT=test
               ARCHIVE=videos.zip; OUT=data/nextqa ;;
  *) echo "unknown dataset" >&2; exit 2 ;;
esac

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE" "$OUT"

docker run --rm -v "$HF_CACHE:/hf" -v "$PWD:/work" -w /work \
  -e HF_HOME=/hf -e DS="$DS" -e CFG="$CFG" -e SPLIT="$SPLIT" \
  -e N="$N" -e OUT="$OUT" -e ARCHIVE="$ARCHIVE" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  python:3.12-slim bash -c '
    set -euo pipefail
    pip install --quiet --disable-pip-version-check --root-user-action=ignore datasets
    python3 tools/build_video_manifest.py --dataset "$DS" --config "$CFG" \
        --split "$SPLIT" --n "$N" --out "$OUT" --video-archive "$ARCHIVE"
    chown -R "$HOST_UID:$HOST_GID" "$OUT" /hf
  '

MANIFEST="$OUT/manifest.jsonl"
[ -s "$MANIFEST" ] || { echo "FAILED: $MANIFEST missing or empty" >&2; exit 1; }
COUNT=$(wc -l < "$MANIFEST"); CLIPS=$(find "$OUT/videos" -type f | wc -l)
echo "ok: $COUNT manifest rows, $CLIPS clips in $OUT"
