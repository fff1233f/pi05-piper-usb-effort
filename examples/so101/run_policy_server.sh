#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_DIR:?Set an OpenPI checkpoint directory, for example .../29999}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVER_PORT="${SERVER_PORT:-8000}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data5/zjh/openpi/jax_cache}"
export HF_HOME="${HF_HOME:-/data5/zjh/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data5/zjh/.cache/uv}"
export TMPDIR="${TMPDIR:-/data5/zjh/openpi/tmp}"

mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${UV_CACHE_DIR}" "${TMPDIR}"

cd "${OPENPI_DIR}"
exec uv run --no-dev scripts/serve_policy.py \
  --port="${SERVER_PORT}" \
  policy:checkpoint \
  --policy.config=pi05_so101_pickup_battery \
  --policy.dir="${CHECKPOINT_DIR}"
