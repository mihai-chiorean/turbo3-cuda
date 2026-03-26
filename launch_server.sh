#!/bin/bash
# launch_server.sh — Launch llama-server with turbo3 KV cache
#
# Usage: ./launch_server.sh [model_path] [context_size]
#
# Environment variables:
#   TURBO_MODEL_PATH  — Path to GGUF model (default: auto-detect in models/)
#   TURBO_CONTEXT     — Context size (default: 524288)
#   TURBO_PORT        — Server port (default: 8080)
#   TURBO_GPU_LAYERS  — GPU layers (default: 99)
#   CUDA_VISIBLE_DEVICES — GPU selection (default: system default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-${TURBO_MODEL_PATH:-$(find "$SCRIPT_DIR/models" -name "*.gguf" -print -quit 2>/dev/null || true)}}"
CONTEXT="${2:-${TURBO_CONTEXT:-524288}}"
PORT="${TURBO_PORT:-8080}"
NGL="${TURBO_GPU_LAYERS:-99}"

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "Error: No model found. Set TURBO_MODEL_PATH or pass as argument."
    echo "Usage: $0 <model.gguf> [context_size]"
    exit 1
fi

echo "Starting turbo3 server:"
echo "  Model:   $MODEL_PATH"
echo "  Context: $CONTEXT tokens"
echo "  Port:    $PORT"
echo "  GPU:     $NGL layers"

"$SCRIPT_DIR/build/bin/llama-server" \
    -m "$MODEL_PATH" \
    --cache-type-k turbo3 \
    --cache-type-v turbo3 \
    -c "$CONTEXT" \
    --port "$PORT" \
    --host 0.0.0.0 \
    -ngl "$NGL" \
    --no-mmap \
    --jinja \
    --reasoning-format deepseek
