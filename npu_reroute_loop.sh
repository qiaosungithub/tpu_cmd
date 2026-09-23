#!/bin/bash
# Standalone RECONCILE-ONLY loop for lyy's npu queue (tmux session npu-reroute).
#
# ★2026-09-16: REROUTE REMOVED (per operator: "不要 reroute, 就用 npu").
# ---------------------------------------------------------------------------
# This loop used to run `route_check --reroute_loop`, whose reroute pass cancels
# a SUBMITTED job that has been PENDING past a deadline and returns it to QUEUED.
# The next tick re-submits it under a NEW xid -- and qwen's checkpoint dir is
# namespaced by xid (CHECKPOINT_BUCKET = xid_<xid>_..., and qwen main.py reads
# only CHECKPOINT_BUCKET, never LOAD_FROM), so the job COLD-STARTS from step 0.
# On 2026-09-16 that cost ~188k steps across ~15 arms (~700-950 GPU-hours).
# A `--reroute_after_s=999999999` flag did NOT stop it (the running binary
# ignored the flag at runtime), so the fix is structural: this loop no longer
# reroutes AT ALL. Preemption recovery now relies on Borg-native same-xid
# restart (max_task_evictions=-1), which resumes because CHECKPOINT_BUCKET is
# frozen per xid.
#
# What this loop does now: ONLY `route_check --reconcile`, i.e. reconcile the
# route queue against XManager truth -- zombie->FAILED, completed->DONE,
# promote SUBMITTED->RUNNING. run_reconcile() contains NO cancel / reroute /
# kill / dequeue path (verified by reading the source 2026-09-16), so it can
# clean the board but can never move a job or drop a checkpoint.
# ★DO NOT re-add --reroute_loop or --reroute here.
#
# ONE-INSTANCE: flock on /tmp/npu-reroute-loop.lock (lock name kept for
# continuity with bootstrap_daemons.sh / the npu-reroute tmux session).
#
# NOT wrapped in `systemd-run --scope`: the old version was, and --scope
# DETACHES route_check from the tmux process tree, so killing the pane left an
# orphan process still holding the flock (burned 2026-09-16 06:xx). Running the
# binary directly makes it a child of this loop, so it dies when the loop dies.
# Memory is bounded by construction instead of by MemoryMax: each cycle is a
# FRESH short-lived one-shot that exits before the next, so nothing accumulates.
BIN='/usr/local/google/_blaze_qiaos/c99224759024385897e236938d1772c2_buildrabbit/execroot/google3/blaze-out/k8-fastbuild/bin/experimental/users/qiaos/tpu_utils/route_check'
QUEUE="$HOME/lyy-work/.npu_local_queue.json"
LOG="$HOME/work/.monitor_watch/npu_reroute_loop.log"
POLL_S=120

exec 9>/tmp/npu-reroute-loop.lock
flock -n 9 || { echo "[npu-reconcile-loop] another instance holds the lock; exiting."; exit 0; }

echo "$(date -u +%FT%TZ)   [reconcile-loop] STARTED (reconcile-only; reroute removed 2026-09-16). poll ${POLL_S}s." >> "$LOG"

while true; do
  if [ ! -x "$BIN" ]; then
    echo "$(date -u +%FT%TZ)   [reconcile-loop] BIN missing/not-exec ($BIN); retry in ${POLL_S}s" >> "$LOG"
    sleep "$POLL_S"
    continue
  fi
  "$BIN" --reconcile --queue_file="$QUEUE" --nodry_run 2>&1 \
    | while IFS= read -r _line; do
        printf '%s   [reconcile-loop] %s\n' "$(date -u +%FT%TZ)" "$_line"
      done >> "$LOG"
  sleep "$POLL_S"
done
