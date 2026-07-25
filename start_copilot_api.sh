#!/usr/bin/env bash
#
# start_copilot_api.sh
#
# Launch caozhiyuan's @jeffreycao/copilot-api gateway inside a persistent tmux
# session so Claude Code can keep talking to the GitHub Copilot API even after
# you close VS Code or drop your SSH/terminal session.
#
# This package requires Node >= 20. The system Node is 18, so we run it from a
# dedicated conda env named "copilotapi" (Node 22+) by putting that env's bin
# directory first on PATH.
#
# Behaviour (idempotent):
#   - If the tmux session already exists -> leave it untouched and exit 0.
#   - If it does not exist               -> create it (detached) and start the
#                                            gateway inside an auto-restart loop.
#
# Usage:
#   ./start_copilot_api.sh                 # start (no-op if already running)
#   tmux attach -t copilot-api             # watch the live logs
#   #   (detach again with: Ctrl-b then d)
#   tmux kill-session -t copilot-api       # stop it

set -euo pipefail

SESSION="copilot-api"
HOST="127.0.0.1"
PORT="4141"

# Node 22+ lives in the dedicated conda env; put it first on PATH.
ENV_BIN="/home/t-qimhuang/miniconda3/envs/copilotapi/bin"
if [[ ! -x "${ENV_BIN}/node" ]]; then
  echo "[copilot-api] ERROR: Node not found at ${ENV_BIN}/node." >&2
  echo "                 Create it with: conda create -n copilotapi -c conda-forge 'nodejs>=22' -y" >&2
  exit 1
fi

# Optional: set API_DEBUG=1 to start with verbose logging.
VERBOSE_FLAG=""
if [[ "${API_DEBUG:-0}" != "0" ]]; then
  VERBOSE_FLAG="--verbose"
fi

# 1) If the session already exists, keep it exactly as it is.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[copilot-api] tmux session '$SESSION' already exists - keeping it as is."
  echo "[copilot-api] attach with: tmux attach -t $SESSION"
  exit 0
fi

# 2) No session yet. Make sure the port is free before spinning one up,
#    otherwise the gateway would fail to bind in an endless restart loop.
if ss -ltn 2>/dev/null | grep -qE "[:.]${PORT}([^0-9]|$)"; then
  echo "[copilot-api] ERROR: port ${PORT} is already in use by another process," >&2
  echo "                 but there is no tmux session named '$SESSION'." >&2
  echo "                 Free the port (or adopt that process) and re-run." >&2
  echo "                 Not starting a duplicate." >&2
  exit 1
fi

# 3) Create the detached session and start the gateway with an auto-restart loop.
echo "[copilot-api] creating tmux session '$SESSION' and starting gateway on ${HOST}:${PORT}..."
tmux new-session -d -s "$SESSION" \
  "export PATH=\"${ENV_BIN}:\$PATH\"; \
   while true; do \
     npx -y @jeffreycao/copilot-api@latest start --port ${PORT} ${VERBOSE_FLAG}; \
     echo \"[copilot-api] gateway exited (code \$?), restarting in 3s...\"; \
     sleep 3; \
   done"

echo "[copilot-api] started."
echo "[copilot-api] attach with: tmux attach -t $SESSION   (detach: Ctrl-b then d)"
echo "[copilot-api] stop with:   tmux kill-session -t $SESSION"
