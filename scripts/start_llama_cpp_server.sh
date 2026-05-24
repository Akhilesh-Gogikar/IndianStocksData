#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/model.gguf [port]" >&2
  exit 2
fi

MODEL_PATH="$1"
PORT="${2:-8080}"
HOST="${LLAMA_CPP_HOST:-127.0.0.1}"
LLAMA_SERVER_BIN="${LLAMA_CPP_SERVER_BIN:-llama-server}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file not found: $MODEL_PATH" >&2
  exit 1
fi

exec "$LLAMA_SERVER_BIN" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH"
