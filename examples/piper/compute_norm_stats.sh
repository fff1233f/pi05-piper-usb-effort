#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/data5/zjh/.cache/uv}"
export TMPDIR="${TMPDIR:-/data5/zjh/openpi/tmp}"
export JAX_PLATFORMS=cpu

mkdir -p "${UV_CACHE_DIR}" "${TMPDIR}"

cd "${OPENPI_DIR}"
exec uv run --no-dev scripts/compute_norm_stats.py --config-name pi05_piper_usb_effort
