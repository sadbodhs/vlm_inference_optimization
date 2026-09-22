#!/usr/bin/env bash
# Continuous RTSP decoders, one container per stream, writing sampled JPEGs to disk.
#
#   docker/run_decoders.sh start 4       # cam1..cam4 at 1 fps
#   docker/run_decoders.sh stop
#
# Decode runs continuously and independently of inference, which is how a real
# deployment works: the camera does not wait for the model. The VLM samples the
# newest frame, and the age of that frame when the answer arrives is the metric
# that matters -- not how fast a single request was served.
set -euo pipefail
cd "$(dirname "$0")/.."

NET="${BENCH_NET:-vlmbench}"
FRAME_DIR="${FRAME_DIR:-/tmp/vlm_frames}"
RTSP_BASE="${RTSP_BASE:-rtsp://127.0.0.1:8554}"
SAMPLE_FPS="${SAMPLE_FPS:-1}"
IMAGE="${DECODER_IMAGE:-linuxserver/ffmpeg:latest}"

case "${1:-start}" in
  start)
    N="${2:-1}"
    mkdir -p "$FRAME_DIR"
    for i in $(seq 1 "$N"); do
      CAM="cam$i"
      rm -rf "${FRAME_DIR:?}/$CAM"; mkdir -p "$FRAME_DIR/$CAM"
      docker rm -f "dec-$CAM" >/dev/null 2>&1 || true
      # --network host: the RTSP server and its publishers already run on the host
      # network namespace, so 127.0.0.1:8554 is where the streams actually are.
      # -update 1 keeps exactly one file per stream: the newest frame, which is the
      # only one a live consumer should ever look at.
      docker run -d --rm --name "dec-$CAM" --network host \
        -v "$FRAME_DIR/$CAM:/out" --entrypoint ffmpeg "$IMAGE" \
        -rtsp_transport tcp -fflags nobuffer -flags low_delay \
        -i "$RTSP_BASE/$CAM" -vf "fps=$SAMPLE_FPS" -q:v 3 \
        -update 1 -y /out/latest.jpg >/dev/null
    done
    echo "started $N decoder(s) at ${SAMPLE_FPS} fps -> $FRAME_DIR/camN/latest.jpg"
    ;;
  stop)
    for c in $(docker ps -q --filter "name=dec-cam"); do docker rm -f "$c" >/dev/null; done
    echo "decoders stopped"
    ;;
  status)
    docker ps --filter "name=dec-cam" --format "  {{.Names}} {{.Status}}"
    for d in "$FRAME_DIR"/cam*/; do
      if [ -f "$d/latest.jpg" ]; then
        echo "  $(basename "$d"): $(stat -c '%y %s' "$d/latest.jpg")"
      else
        echo "  $(basename "$d"): no frame yet"
      fi
    done
    ;;
  *) echo "usage: $0 [start N|stop|status]" >&2; exit 2 ;;
esac
