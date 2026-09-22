#!/usr/bin/env bash
# Deprecated shim. The launcher dispatches on the arm's `stack:` field now, so that
# swapping serving stacks is a YAML edit rather than a different command.
exec "$(dirname "$0")/run_server.sh" "$@"
