#!/usr/bin/env bash
set -euo pipefail

: "${A100_SSH_HOST:?Set A100_SSH_HOST, for example zjh@114.214.255.57}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

exec ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${A100_SSH_HOST}"
