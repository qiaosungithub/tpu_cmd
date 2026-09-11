#!/bin/bash
# Standalone reconcile+reroute loop for lyy's npu queue (tmux npu-reroute).
#
# Same contract as tpu_reroute_loop_v17.sh, pointed at
# ~/lyy-work/.npu_local_queue.json. npu-daemon used to run only --reroute
# (UNKNOWN = no-op), never --reconcile, so XM-completed rows stayed SUBMITTED
# for days. This process owns reconcile+reroute; npu-daemon must set
# TPU_ROUTE_INLANE_REROUTE=0 so the two do not double-write.
#
# ONE-INSTANCE: flock on /tmp/npu-reroute-loop.lock.
# Do NOT wrap the binary in stdbuf (GRTE + LD_PRELOAD dies on glibc).
BIN='/usr/local/google/_blaze_qiaos/c99224759024385897e236938d1772c2_buildrabbit/execroot/google3/blaze-out/k8-fastbuild/bin/experimental/users/qiaos/tpu_utils/route_check'
QUEUE="$HOME/lyy-work/.npu_local_queue.json"
LOG="$HOME/work/.monitor_watch/npu_reroute_loop.log"

exec 9>/tmp/npu-reroute-loop.lock
flock -n 9 || { echo "[npu-reroute-loop] another instance holds the lock; exiting."; exit 0; }

while true; do
  systemd-run --user --scope -q \
      -p MemoryMax=8G -p MemorySwapMax=0 \
      "$BIN" --queue_file="$QUEUE" --reroute_loop --nodry_run 2>&1 \
      | while IFS= read -r _line; do
          printf '%s %s\n' "$(date -u +%FT%TZ)" "$_line"
        done >> "$LOG"
  echo "$(date -u +%FT%TZ) [npu-reroute-loop] loop EXITED rc=$?; restarting in 30s" >> "$LOG"
  sleep 30
done
