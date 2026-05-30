#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/model.gguf [port]" >&2
  echo "Optional env: LLAMA_CPP_SPEC_ENABLE=1 LLAMA_CPP_SPEC_TYPE=draft-mtp LLAMA_CPP_SPEC_DRAFT_N_MAX=3" >&2
  exit 2
fi

MODEL_PATH="$1"
PORT="${2:-8080}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  echo "Invalid port: $PORT. Port must be an integer between 1 and 65535." >&2
  exit 1
fi
HOST="${LLAMA_CPP_HOST:-127.0.0.1}"
LLAMA_SERVER_BIN="${LLAMA_CPP_SERVER_BIN:-llama-server}"
SPEC_ENABLE_RAW="${LLAMA_CPP_SPEC_ENABLE:-0}"
SPEC_TYPE_RAW="${LLAMA_CPP_SPEC_TYPE:-none}"
SPEC_DRAFT_N_MAX_RAW="${LLAMA_CPP_SPEC_DRAFT_N_MAX:-3}"
SPEC_DRAFT_N_MIN_RAW="${LLAMA_CPP_SPEC_DRAFT_N_MIN:-0}"
SPEC_NGRAM_MOD_N_MIN_RAW="${LLAMA_CPP_SPEC_NGRAM_MOD_N_MIN:-}"
SPEC_NGRAM_MOD_N_MAX_RAW="${LLAMA_CPP_SPEC_NGRAM_MOD_N_MAX:-}"
SPEC_NGRAM_MOD_N_MATCH_RAW="${LLAMA_CPP_SPEC_NGRAM_MOD_N_MATCH:-}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file not found: $MODEL_PATH" >&2
  exit 1
fi

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

SPEC_TYPE="$(echo "$SPEC_TYPE_RAW" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
if [[ "$SPEC_TYPE" == "mtp" ]]; then
  echo "warning: LLAMA_CPP_SPEC_TYPE=mtp is deprecated; mapping to draft-mtp" >&2
  SPEC_TYPE="draft-mtp"
fi

ARGS=(
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL_PATH"
)

if is_truthy "$SPEC_ENABLE_RAW"; then
  if [[ "$SPEC_TYPE" == "none" || -z "$SPEC_TYPE" ]]; then
    echo "warning: LLAMA_CPP_SPEC_ENABLE=1 but LLAMA_CPP_SPEC_TYPE is none; launching without speculative decoding" >&2
  elif ! is_non_negative_int "$SPEC_DRAFT_N_MAX_RAW"; then
    echo "warning: LLAMA_CPP_SPEC_DRAFT_N_MAX must be a non-negative integer; launching without speculative decoding" >&2
  elif ! is_non_negative_int "$SPEC_DRAFT_N_MIN_RAW"; then
    echo "warning: LLAMA_CPP_SPEC_DRAFT_N_MIN must be a non-negative integer; launching without speculative decoding" >&2
  else
    ARGS+=(--spec-type "$SPEC_TYPE")
    ARGS+=(--spec-draft-n-max "$SPEC_DRAFT_N_MAX_RAW")
    ARGS+=(--spec-draft-n-min "$SPEC_DRAFT_N_MIN_RAW")
    if [[ -n "$SPEC_NGRAM_MOD_N_MIN_RAW" ]]; then
      if is_non_negative_int "$SPEC_NGRAM_MOD_N_MIN_RAW"; then
        ARGS+=(--spec-ngram-mod-n-min "$SPEC_NGRAM_MOD_N_MIN_RAW")
      else
        echo "warning: ignoring LLAMA_CPP_SPEC_NGRAM_MOD_N_MIN (expected non-negative integer)" >&2
      fi
    fi
    if [[ -n "$SPEC_NGRAM_MOD_N_MAX_RAW" ]]; then
      if is_non_negative_int "$SPEC_NGRAM_MOD_N_MAX_RAW"; then
        ARGS+=(--spec-ngram-mod-n-max "$SPEC_NGRAM_MOD_N_MAX_RAW")
      else
        echo "warning: ignoring LLAMA_CPP_SPEC_NGRAM_MOD_N_MAX (expected non-negative integer)" >&2
      fi
    fi
    if [[ -n "$SPEC_NGRAM_MOD_N_MATCH_RAW" ]]; then
      if is_non_negative_int "$SPEC_NGRAM_MOD_N_MATCH_RAW"; then
        ARGS+=(--spec-ngram-mod-n-match "$SPEC_NGRAM_MOD_N_MATCH_RAW")
      else
        echo "warning: ignoring LLAMA_CPP_SPEC_NGRAM_MOD_N_MATCH (expected non-negative integer)" >&2
      fi
    fi
  fi
fi

exec "$LLAMA_SERVER_BIN" "${ARGS[@]}"
