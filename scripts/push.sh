#!/usr/bin/env bash
# Sync this working copy to the rig and push from there.
#
# The GitHub key lives on the 3090, not on the authoring machine, so pushes have
# to originate there. Syncing the working copy carries .git along with it, which
# overwrites the rig's remote configuration -- so the remote is re-established
# every time rather than assumed. That has bitten three times; this makes it
# idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

RIG="${RIG:-3090}"
RIG_PATH="${RIG_PATH:-~/sadbodh/vlm_inference_optimization}"
REMOTE="${REMOTE:-git@github.com:sadbodhs/vlm_inference_optimization.git}"

# results/ and data/ flow ONE way: rig -> here. They are produced on the rig, and
# rsync overwrites on any difference, not on age -- so syncing them outward lets a
# stale local copy silently replace a fresh measurement. That happened: a corrected
# E1 sweep (88.7% of roofline) was overwritten on the rig by the pre-correction
# local copy (92.5%), and the published figure kept showing the wrong number.
# Commits still carry results: the objects travel in .git, not the working tree.
echo "syncing -> $RIG:$RIG_PATH"
rsync -az --exclude '__pycache__' --exclude '.DS_Store' \
      --exclude '_site' --exclude 'site_src' \
      --exclude '/results/' --exclude '/data/' \
      ./ "$RIG:$RIG_PATH/"

ssh -o BatchMode=yes "$RIG" "
  set -euo pipefail
  cd $RIG_PATH
  git remote add origin '$REMOTE' 2>/dev/null || git remote set-url origin '$REMOTE'
  echo '  remote: '\$(git remote get-url origin)
  echo '  head:   '\$(git log --oneline -1)
  git push origin main
"
