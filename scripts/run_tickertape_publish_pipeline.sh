#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="daily"
DRY_RUN=0
SKIP_UPLOAD=0
REQUIRE_GATE_PASS="${REQUIRE_GATE_PASS:-0}"
WAIT_LOCK_SECONDS="${LOCK_WAIT_SECONDS:-0}"
SNAPSHOT_DATE="$(date -u +%F)"
COMPANY_LIST="full-company-list.json"
DB_PATH="local_repository/tickertape.sqlite"
RAW_DIR="local_repository/raw"
TARGET_DB="system/market_intel.db"
SCHEMA_PATH="system/schema.sql"
LOGS_DIR="local_repository/logs"
MIN_SUCCESS_RATE="0.98"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SYNC_RETRIES="${SYNC_RETRIES:-2}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-20}"
SYNC_WORKERS="${SYNC_WORKERS:-3}"
INCLUDE_RAW="${INCLUDE_RAW:-0}"
HOST="${HOST:-NewBlogProject-Server}"
DB_NAME="${DB_NAME:-cerebral_insights}"
SYNC_EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_tickertape_publish_pipeline.sh [options] [-- extra tickertape_sync args]

Options:
  --mode MODE              daily, sync-only, repair, gate-only, publish-only
  --snapshot-date DATE     UTC snapshot date, default today
  --company-list PATH      TickerTape company list JSON
  --db PATH                local TickerTape SQLite database
  --raw-dir PATH           local raw payload directory
  --target-db PATH         canonical serving SQLite database
  --schema PATH            canonical schema file
  --logs-dir PATH          pipeline log/output directory
  --min-success-rate RATE  publish gate threshold, default 0.98
  --progress-every N       sync progress cadence
  --sync-retries N         sync retry count
  --sync-timeout N         sync request timeout seconds
  --sync-workers N         sync fetch worker count, default 3
  --wait-lock SECONDS      wait for another pipeline run before exiting
  --skip-upload            run through canonicalization but do not upload
  --require-gate-pass      fail pipeline when publish gate fails
  --dry-run                print commands without running them
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --snapshot-date) SNAPSHOT_DATE="$2"; shift 2 ;;
    --company-list) COMPANY_LIST="$2"; shift 2 ;;
    --db) DB_PATH="$2"; shift 2 ;;
    --raw-dir) RAW_DIR="$2"; shift 2 ;;
    --target-db) TARGET_DB="$2"; shift 2 ;;
    --schema) SCHEMA_PATH="$2"; shift 2 ;;
    --logs-dir) LOGS_DIR="$2"; shift 2 ;;
    --min-success-rate) MIN_SUCCESS_RATE="$2"; shift 2 ;;
    --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
    --sync-retries) SYNC_RETRIES="$2"; shift 2 ;;
    --sync-timeout) SYNC_TIMEOUT="$2"; shift 2 ;;
    --sync-workers) SYNC_WORKERS="$2"; shift 2 ;;
    --wait-lock) WAIT_LOCK_SECONDS="$2"; shift 2 ;;
    --skip-upload) SKIP_UPLOAD=1; shift ;;
    --require-gate-pass) REQUIRE_GATE_PASS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; SYNC_EXTRA_ARGS+=("$@"); break ;;
    *) SYNC_EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$MODE" in
  daily|sync-only|repair|gate-only|publish-only) ;;
  *) echo "Unknown mode: $MODE" >&2; usage >&2; exit 64 ;;
esac

if [[ ! "$SYNC_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --sync-workers value: $SYNC_WORKERS" >&2
  exit 64
fi

mkdir -p "$LOGS_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG="$LOGS_DIR/tickertape_publish_pipeline_${MODE}_${STAMP}.log"
MANIFEST_PATH="$LOGS_DIR/tickertape_publish_manifest_${STAMP}.json"
FAILURE_REPORT_PATH="$LOGS_DIR/tickertape_publish_failures_${STAMP}.json"

ln -sf "$(basename "$PIPELINE_LOG")" "$LOGS_DIR/tickertape_publish_pipeline_latest.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "tickertape_publish_pipeline mode=$MODE snapshot_date=$SNAPSHOT_DATE started_at=$(date -u +%FT%TZ)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "$ROOT_DIR/.venv/bin/python3" \
    "$HOME/miniconda3/bin/python3" \
    "$(command -v python3)"
  do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No Python interpreter found. Set PYTHON_BIN."
  exit 1
fi
export PYTHON_BIN
echo "python_bin=$PYTHON_BIN"

LOCK_DIR="$ROOT_DIR/local_repository/.tickertape_publish_pipeline.lock"

acquire_lock() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry_run=1; skipping lock acquisition"
    return 0
  fi
  local waited=0
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    if [[ -f "$LOCK_DIR/pid" ]]; then
      local pid
      pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        echo "removing stale lock for pid=$pid"
        rm -rf "$LOCK_DIR"
        continue
      fi
    fi
    if (( waited >= WAIT_LOCK_SECONDS )); then
      echo "Another TickerTape publish pipeline is running: $LOCK_DIR"
      exit 75
    fi
    echo "Waiting for pipeline lock: waited=${waited}s limit=${WAIT_LOCK_SECONDS}s"
    sleep 30
    waited=$((waited + 30))
  done
  echo "$$" > "$LOCK_DIR/pid"
  trap 'rm -rf "$LOCK_DIR"' EXIT
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_sync_pass() {
  local label="$1"
  shift
  local cmd=(
    bash scripts/run_tickertape_daily_sync.sh
    --company-list "$COMPANY_LIST"
    --db "$DB_PATH"
    --raw-dir "$RAW_DIR"
    --snapshot-date "$SNAPSHOT_DATE"
    --progress-every "$PROGRESS_EVERY"
    --retries "$SYNC_RETRIES"
    --timeout "$SYNC_TIMEOUT"
    --workers "$SYNC_WORKERS"
    "$@"
  )
  if (( ${#SYNC_EXTRA_ARGS[@]} > 0 )); then
    cmd+=("${SYNC_EXTRA_ARGS[@]}")
  fi
  echo "sync_pass=$label"
  print_cmd "${cmd[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  set +e
  "${cmd[@]}"
  local status=$?
  set -e
  if [[ "$status" == "0" || "$status" == "2" ]]; then
    echo "sync_pass=$label exit_status=$status continuing"
    return 0
  fi
  echo "sync_pass=$label exit_status=$status stopping"
  return "$status"
}

run_status() {
  run_cmd "$PYTHON_BIN" tools/tickertape_status.py --db "$DB_PATH"
}

run_gate() {
  local cmd=(
    "$PYTHON_BIN" tools/tickertape_publish_gate.py
    --db "$DB_PATH"
    --company-list "$COMPANY_LIST"
    --snapshot-date "$SNAPSHOT_DATE"
    --min-success-rate "$MIN_SUCCESS_RATE"
    --manifest "$MANIFEST_PATH"
    --failure-report "$FAILURE_REPORT_PATH"
  )
  print_cmd "${cmd[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  set +e
  "${cmd[@]}"
  local status=$?
  set -e
  ln -sf "$(basename "$MANIFEST_PATH")" "$LOGS_DIR/tickertape_publish_manifest_latest.json"
  ln -sf "$(basename "$FAILURE_REPORT_PATH")" "$LOGS_DIR/tickertape_publish_failures_latest.json"
  if [[ "$status" != "0" ]]; then
    echo "publish_gate_blocked exit_status=$status manifest=$MANIFEST_PATH failure_report=$FAILURE_REPORT_PATH"
    if [[ "$status" != "2" ]]; then
      return "$status"
    fi
    if [[ "$REQUIRE_GATE_PASS" == "1" || "$MODE" == "gate-only" ]]; then
      return "$status"
    fi
    echo "publish_gate_override mode=$MODE require_gate_pass=$REQUIRE_GATE_PASS continuing_with_upload=1"
    return 0
  fi
  echo "publish_gate_passed manifest=$MANIFEST_PATH"
}

run_canonicalize() {
  run_cmd "$PYTHON_BIN" system/canonical_tickertape.py \
    --source-db "$DB_PATH" \
    --target-db "$TARGET_DB" \
    --schema "$SCHEMA_PATH"
}

run_upload() {
  if [[ "$SKIP_UPLOAD" == "1" ]]; then
    echo "skip_upload=1; server upload skipped"
    return 0
  fi
  run_cmd env INCLUDE_RAW="$INCLUDE_RAW" HOST="$HOST" DB_NAME="$DB_NAME" \
    bash cerebral-insights-platform/scripts/load_tickertape_to_server.sh
}

acquire_lock

case "$MODE" in
  daily)
    run_sync_pass "primary"
    run_sync_pass "repair"
    run_status
    run_gate
    run_canonicalize
    run_upload
    ;;
  sync-only)
    run_sync_pass "primary"
    run_status
    ;;
  repair)
    run_sync_pass "repair"
    run_status
    run_gate
    run_canonicalize
    run_upload
    ;;
  gate-only)
    run_status
    run_gate
    ;;
  publish-only)
    run_status
    run_gate
    run_canonicalize
    run_upload
    ;;
esac

echo "tickertape_publish_pipeline mode=$MODE finished_at=$(date -u +%FT%TZ)"
