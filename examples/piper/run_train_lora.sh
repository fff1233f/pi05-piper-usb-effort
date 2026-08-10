#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="pi05_piper_usb_effort_lora"
export EXP_NAME="${EXP_NAME:-usb_effort_dual_lora_bs32_30k}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-60000}"

exec "${SCRIPT_DIR}/run_train.sh"
