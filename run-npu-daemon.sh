#!/usr/bin/env bash
# The check-board daemon for lyy's registry (the `npu` half of tpu_wrapper.sh).
#
#   tmux new-session -d -s npu-daemon -c ~/work/tpu_cmd \
#     'while true; do bash run-npu-daemon.sh; echo "restarting in 5s"; sleep 5; done'
#
# `npu check` renders from $TPU_CHECK_CACHE_FILE. Without this process nothing
# ever writes that file, so every one of lyy's jobs reads SUBMITTED forever no
# matter what it is really doing — the board looks alive and is not.
#
# This is the SAME script as sqa's daemon, pointed at lyy's three files. Do not
# copy the daemon; a second copy is a second thing to keep in sync.
set -euo pipefail
cd "$(dirname "$0")"

export TPU_JOBS_FILE="${NPU_JOBS_FILE:-$HOME/lyy-work/.npu_jobs.json}"
export TPU_CHECK_CACHE_FILE="${NPU_CHECK_CACHE_FILE:-$HOME/lyy-work/.npu_check_cache.txt}"
export TPU_CHECK_TIME_FILE="${NPU_CHECK_TIME_FILE:-$HOME/lyy-work/.npu_check_time.txt}"

exec bash "$PWD/tpu_check_daemon.sh"
