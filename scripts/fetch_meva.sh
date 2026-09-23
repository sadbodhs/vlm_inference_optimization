#!/usr/bin/env bash
# Download the E7 clips listed in results/e7/selection.json from MEVA's public
# bucket (anonymous; CC BY 4.0), then probe each so a truncated download fails
# here rather than as a short clip in the middle of an experiment.
#
# MAX_BW caps the transfer (e.g. MAX_BW=25MB/s). Off by default: the two rig
# network drops that first looked transfer-induced were traced to an IP conflict
# between the rig's own Wi-Fi and Ethernet interfaces (NetworkManager "conflict
# detected" at a DHCP renewal), not to transfer load.
set -euo pipefail
cd "$(dirname "$0")/.."
SEL="${SEL:-results/e7/selection.json}"
OUT=data/meva/video
MAX_BW="${MAX_BW:-0}"
mkdir -p "$OUT"

CFG=$(mktemp -d)
printf '[default]\nregion = us-east-1\n' > "$CFG/config"
[ "$MAX_BW" != "0" ] && printf 's3 =\n  max_bandwidth = %s\n  max_concurrent_requests = 2\n' "$MAX_BW" >> "$CFG/config"

python3 -c "import json; [print(c['s3_key'], c['bytes']) for c in json.load(open('$SEL'))['clips']]" |
while read -r key bytes; do
  f="$OUT/$(basename "$key")"
  # skip only an exact-size file: an interrupted copy leaves a partial one behind
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$bytes" ]; then echo "  have $(basename "$f")"; continue; fi
  rm -f "$f"
  docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/$OUT:/out" -v "$CFG:/cfg:ro" \
    -e AWS_CONFIG_FILE=/cfg/config amazon/aws-cli \
    s3 cp --no-sign-request --only-show-errors "s3://mevadata-public-01/$key" "/out/$(basename "$key")"
  [ "$(stat -c %s "$f")" = "$bytes" ] || { echo "  SIZE MISMATCH $(basename "$f")"; exit 1; }
  echo "  got  $(basename "$f")"
done
rm -rf "$CFG"

echo "--- probe ---"
bad=0
for f in "$OUT"/*.avi; do
  line=$(docker run --rm -v "$PWD/$OUT:/v" --entrypoint ffprobe linuxserver/ffmpeg:latest \
    -v error -select_streams v:0 -count_packets \
    -show_entries stream=width,height,r_frame_rate,nb_read_packets -of csv=p=0 \
    "/v/$(basename "$f")")
  n=$(echo "$line" | cut -d, -f4)
  printf "  %-60s %s\n" "$(basename "$f")" "$line"
  [ "${n:-0}" -ge 8700 ] || { echo "    SHORT: $n packets"; bad=1; }
done
[ "$bad" = 0 ] && echo "all clips complete"
