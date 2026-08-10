#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/so101_client.py" \
  --host="${SERVER_HOST:-127.0.0.1}" \
  --port="${SERVER_PORT:-8000}" \
  --robot-port="${SO101_PORT:-/dev/ttyACM0}" \
  --robot-id="${SO101_ID:-zjh_follower_arm}" \
  --calibration-dir="${SO101_CALIBRATION_DIR:-${HOME}/.cache/huggingface/lerobot/calibration/robots/so_follower}" \
  --front-camera="${FRONT_CAMERA_INDEX:-0}" \
  --wrist-camera="${WRIST_CAMERA_INDEX:-2}" \
  --camera-width="${CAMERA_WIDTH:-1280}" \
  --camera-height="${CAMERA_HEIGHT:-720}" \
  --camera-fps="${CAMERA_FPS:-30}" \
  --camera-fourcc="${CAMERA_FOURCC:-MJPG}" \
  --control-fps="${CONTROL_FPS:-30}" \
  --actions-per-chunk="${ACTIONS_PER_CHUNK:-50}" \
  --max-relative-target="${SO101_MAX_RELATIVE_TARGET:-5}" \
  --latency-log-dir="${SO101_LOG_DIR:-${HOME}/openpi/so101_logs}" \
  --task="${TASK:-Grab blue battery to the bin}"
