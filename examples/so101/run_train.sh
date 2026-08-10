#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set exactly two GPU ids, for example 0,1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_NAME="${EXP_NAME:-full_sft_30k_bs18_fsdp2}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-60000}"

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#gpu_ids[@]}" -ne 2 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly two GPU ids." >&2
  exit 1
fi

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
    echo "Both GPUs must have the same model; got ${gpu_model} and ${model}." >&2
    exit 1
  fi
  gpu_model="${model}"
  if (( free_mib < MIN_GPU_FREE_MIB )); then
    echo "GPU ${gpu_id} has ${free_mib} MiB free; at least ${MIN_GPU_FREE_MIB} MiB is required." >&2
    exit 1
  fi
done

export HF_HOME="${HF_HOME:-/data5/zjh/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data5/zjh/.cache/uv}"
export WANDB_DIR="${WANDB_DIR:-/data5/zjh/wandb/openpi/runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/data5/zjh/wandb/openpi/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/data5/zjh/wandb/openpi/data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data5/zjh/openpi/jax_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export TMPDIR="${TMPDIR:-/data5/zjh/openpi/tmp}"

mkdir -p "${UV_CACHE_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" \
  "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}"

extra_args=()
if [[ "${RESUME:-false}" == "true" ]]; then
  extra_args+=(--resume)
elif [[ "${OVERWRITE:-false}" == "true" ]]; then
  extra_args+=(--overwrite)
fi

cd "${OPENPI_DIR}"
exec uv run --no-dev scripts/train.py pi05_so101_pickup_battery \
  --exp-name="${EXP_NAME}" \
  "${extra_args[@]}"
