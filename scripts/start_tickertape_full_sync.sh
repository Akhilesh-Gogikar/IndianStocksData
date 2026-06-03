#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p local_repository/logs

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LAUNCH_LOG="local_repository/logs/tickertape_full_sync_launch_${STAMP}.log"
SESSION_NAME="tickertape_full_sync_${STAMP}"
RUN_CMD=(bash scripts/run_tickertape_publish_pipeline.sh)
if [[ $# -eq 0 ]]; then
  RUN_CMD+=(--mode daily)
else
  RUN_CMD+=("$@")
fi

if command -v screen >/dev/null 2>&1; then
  screen -dmS "$SESSION_NAME" bash -lc 'cd "$1" && shift && exec "$@"' bash "$ROOT_DIR" "${RUN_CMD[@]}"
  echo "started launcher=screen session=$SESSION_NAME"
else
  nohup "${RUN_CMD[@]}" > "$LAUNCH_LOG" 2>&1 &
  PID="$!"
  echo "started launcher=nohup pid=$PID"
fi

echo "launch_log=$ROOT_DIR/$LAUNCH_LOG"
echo "latest_pipeline_log=$ROOT_DIR/local_repository/logs/tickertape_publish_pipeline_latest.log"
echo "latest_sync_log=$ROOT_DIR/local_repository/logs/tickertape_sync_latest.log"
echo "database=$ROOT_DIR/local_repository/tickertape.sqlite"
