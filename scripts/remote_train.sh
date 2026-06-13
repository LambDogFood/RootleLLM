#!/usr/bin/env bash
# Sync this project to a training PC on the LAN, then run a command there over SSH.
# Develop on the Mac; train on the PC's GPU (inside Docker / WSL2). Code is pushed;
# data/ and out/ stay on the PC (excluded from the sync, bind-mounted into Docker).
#
# Required env (set once per shell):
#   export ROOTLLM_REMOTE=alex@192.168.1.50     # user@host of the PC
#   export ROOTLLM_REMOTE_PORT=2222             # SSH port (WSL2 uses 2222 in our setup)
# Optional:
#   export ROOTLLM_REMOTE_DIR=RootlLLM          # path on the PC, in the WSL2 home fs
#   export ROOTLLM_REMOTE_PREP="source ~/venv/bin/activate"   # run before the command
#
# Usage (the default command trains via Docker):
#   scripts/remote_train.sh "docker compose build"            # one-time / after dep changes
#   scripts/remote_train.sh "docker compose run --rm gpu-check"   # verify the GPU is visible
#   scripts/remote_train.sh "docker compose run --rm prep"    # download + tokenise data
#   scripts/remote_train.sh                                   # default: train on the GPU
#   scripts/remote_train.sh "docker compose run --rm train"   # same, explicit
#   scripts/remote_train.sh "python -m pytest"                # run tests on the PC (no Docker)
set -euo pipefail

REMOTE="${ROOTLLM_REMOTE:?set ROOTLLM_REMOTE=user@host (e.g. alex@192.168.1.50)}"
PORT="${ROOTLLM_REMOTE_PORT:-22}"
REMOTE_DIR="${ROOTLLM_REMOTE_DIR:-RootlLLM}"
PREP="${ROOTLLM_REMOTE_PREP:-}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

CMD="${1:-docker compose run --rm train}"

echo ">> syncing code to $REMOTE:$REMOTE_DIR (port $PORT) ..."
rsync -avz --delete -e "ssh -p $PORT" \
  --exclude '.git' --exclude 'out/' --exclude 'data/' \
  --exclude '__pycache__/' --exclude '*.egg-info' \
  --exclude '.pytest_cache' --exclude '.ruff_cache' --exclude '.venv' \
  "$HERE/" "$REMOTE:$REMOTE_DIR/"

echo ">> running on $REMOTE: $CMD"
ssh -t -p "$PORT" "$REMOTE" "cd '$REMOTE_DIR' && ${PREP:+$PREP && }$CMD"
