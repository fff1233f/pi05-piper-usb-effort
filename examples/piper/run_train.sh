#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set one or two GPU ids, for example 1 or 0,1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_NAME="${EXP_NAME:-usb_effort_v1}"
CONFIG_NAME="${CONFIG_NAME:-pi05_piper_usb_effort}"

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
gpu_count="${#gpu_ids[@]}"
if [[ "${gpu_count}" -ne 1 && "${gpu_count}" -ne 2 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain one or two GPU ids." >&2
  exit 1
fi

if [[ "${gpu_count}" -eq 1 ]]; then
  BATCH_SIZE="${BATCH_SIZE:-4}"
  MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-76000}"
  DEFAULT_MEMORY_FRACTION="0.95"
else
  BATCH_SIZE="${BATCH_SIZE:-24}"
  MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-72000}"
  DEFAULT_MEMORY_FRACTION="0.90"
fi
FSDP_DEVICES="${gpu_count}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"

gpu_model=""
for gpu_id in "${gpu_ids[@]}"; do
  gpu_info="$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v id="${gpu_id}" '$1 + 0 == id {gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); print $2 "|" $3}')"
  if [[ -z "${gpu_info}" ]]; then
    echo "GPU ${gpu_id} was not found." >&2
    exit 1
  fi
  model="${gpu_info%%|*}"
  free_mib="${gpu_info##*|}"
  if [[ -n "${gpu_model}" && "${model}" != "${gpu_model}" ]]; then
    echo "All selected GPUs must have the same model; got ${gpu_model} and ${model}." >&2
    exit 1
  fi
  gpu_model="${model}"
  if (( free_mib < MIN_GPU_FREE_MIB )); then
    echo "GPU ${gpu_id} has ${free_mib} MiB free; batch ${BATCH_SIZE} requires at least ${MIN_GPU_FREE_MIB} MiB." >&2
    exit 1
  fi
done

export UV_CACHE_DIR="${UV_CACHE_DIR:-/data5/zjh/.cache/uv}"
export WANDB_DIR="${WANDB_DIR:-/data5/zjh/wandb/openpi/runs}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data5/zjh/openpi/jax_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-${DEFAULT_MEMORY_FRACTION}}"
export TMPDIR="${TMPDIR:-/data5/zjh/openpi/tmp}"

mkdir -p "${UV_CACHE_DIR}" "${WANDB_DIR}" "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}"

extra_args=()
if [[ "${RESUME:-false}" == "true" ]]; then
  extra_args+=(--resume)
elif [[ "${OVERWRITE:-false}" == "true" ]]; then
  extra_args+=(--overwrite)
fi

cd "${OPENPI_DIR}"
exec uv run --no-dev scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --batch-size="${BATCH_SIZE}" \
  --fsdp-devices="${FSDP_DEVICES}" \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  "${extra_args[@]}"
