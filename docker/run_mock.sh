#!/usr/bin/env bash
# Mock VLM server in a container, for GPU-free dry runs.
#   docker/run_mock.sh start | stop
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${HARNESS_IMAGE:-vlmbench-harness:latest}"
NET="${BENCH_NET:-vlmbench}"
NAME="${MOCK_NAME:-vlm-mock}"
PORT="${MOCK_PORT:-8077}"

case "${1:-start}" in
  start)
    docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --rm --name "$NAME" --network "$NET" \
      -v "$PWD:/work" -w /work -p "127.0.0.1:$PORT:$PORT" \
      "$IMAGE" python3 tools/mock_vlm_server.py --port "$PORT" \
        --max-running "${MOCK_BATCH:-8}" --itl-ms "${MOCK_ITL:-12}" >/dev/null
    echo "mock server: http://$NAME:$PORT (in-network) / http://127.0.0.1:$PORT (host)"
    ;;
  stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" ;;
  *) echo "usage: $0 [start|stop]" >&2; exit 2 ;;
esac
