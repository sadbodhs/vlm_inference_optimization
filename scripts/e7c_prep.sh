#!/usr/bin/env bash
# E7c live-run camera files: for every (clip, start) the requested camera counts
# need, stream-copy the clip from that keyframe and loop it once, so camera i plays
# exactly what E7b's camera i saw. MEVA's GOP is 60 frames, so every E7b start
# (a multiple of 60) is a keyframe and no re-encoding is needed. MEVA's AVIs carry
# no timestamps and Matroska refuses a packet without one, hence +genpts.
#   CAMS=4,6,8,10,12 scripts/e7c_prep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CAMS="${CAMS:-4,6,8,10,12,14}"
T=data/meva/trim; mkdir -p "$T"
FF="docker run --rm -u $(id -u):$(id -g) -v $PWD/data/meva:/d --entrypoint"

# real frame counts, once
if [ ! -s "$T/nframes.json" ]; then
  echo "{" > "$T/nframes.json.tmp"
  first=1
  for f in data/meva/video/*.avi; do
    c=$(basename "$f" .r13.avi)
    n=$($FF ffprobe linuxserver/ffmpeg:latest -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets -of csv=p=0 "/d/video/$(basename "$f")")
    [ $first = 1 ] || echo "," >> "$T/nframes.json.tmp"; first=0
    printf '"%s": %s' "$c" "$n" >> "$T/nframes.json.tmp"
  done
  echo "}" >> "$T/nframes.json.tmp"; mv "$T/nframes.json.tmp" "$T/nframes.json"
fi

docker/run_harness.sh python3 experiments/e7c.py prep --cams "$CAMS" | while read -r clip k; do
  out="$T/${clip}_${k}.mkv"
  [ -s "$out" ] && continue
  printf "file '/d/video/%s.r13.avi'\ninpoint %s\nfile '/d/video/%s.r13.avi'\n" \
    "$clip" "$(python3 -c "print(f'{$k/30:.6f}')")" "$clip" > "$T/${clip}_${k}.txt"
  $FF ffmpeg linuxserver/ffmpeg:latest -v error -y -fflags +genpts -f concat -safe 0 -i "/d/trim/${clip}_${k}.txt" \
     -c copy -t 145 "/d/trim/${clip}_${k}.mkv"
  echo "  cut $clip from $k"
done

# alignment check: frame 0 of each camera file must match the E7 JPEG of frame k
# better than the JPEGs one window either side (a static scene can tie; report it)
echo "--- alignment ---"
for m in "$T"/*.mkv; do
  b=$(basename "$m" .mkv); clip=${b%_*}; k=${b##*_}
  $FF ffmpeg linuxserver/ffmpeg:latest -v error -y -i "/d/trim/$b.mkv" -frames:v 1 "/d/trim/$b.f0.png"
  best=""; bestv=0
  for d in -60 0 60; do
    j=$((k + d)); [ $j -lt 0 ] && continue
    jp=$(printf "/d/frames5/%s/%05d.jpg" "$clip" $j)
    [ -f "data/meva/frames5/$clip/$(printf %05d $j).jpg" ] || continue
    v=$($FF ffmpeg linuxserver/ffmpeg:latest -v info -i "/d/trim/$b.f0.png" -i "$jp" -lavfi psnr -f null - 2>&1 \
        | sed -n 's/.*average:\([0-9.inf]*\).*/\1/p' | tail -1)
    [ "$v" = "inf" ] && v=99
    if python3 -c "import sys; sys.exit(0 if float('${v:-0}') > $bestv else 1)"; then bestv=$v; best=$d; fi
  done
  printf "  %-55s best match at offset %+d (PSNR %s dB)\n" "$b" "$best" "$bestv"
  rm -f "$T/$b.f0.png"
done
