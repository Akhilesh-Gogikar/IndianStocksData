#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p local_repository/logs

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="local_repository/logs/tickertape_sync_${STAMP}.log"

ln -sf "$(basename "$LOG_PATH")" local_repository/logs/tickertape_sync_latest.log

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "$ROOT_DIR/.venv/bin/python3" \
    "$HOME/miniconda3/bin/python3" \
    "$(command -v python3)"
  do
    if [[ -x "$candidate" ]] && "$candidate" -c "import requests" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No Python interpreter with requests installed. Set PYTHON_BIN or install requests."
  exit 1
fi

echo "python_bin=$PYTHON_BIN"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" tools/tickertape_sync.py \
  --company-list full-company-list.json \
  --db local_repository/tickertape.sqlite \
  --raw-dir local_repository/raw \
  "$@" 2>&1 | tee "$LOG_PATH"
