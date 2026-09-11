#!/bin/bash
# Scheduler <-> rerouter INTERLOCK (operator order, 2026-08-30 21:34Z):
#   "调度器开的时候随机检测 rerouter 有没有开,如果没开自动开,否则不运转。"
#
# Each tick (randomised interval, see SLEEP below):
#   1. Is the rerouter alive?  -> ps by pid-file, NOT pgrep -f (a -f pattern
#      matches this script's own argv; FLEET_STANDING records three such
#      self-matches in one night).
#   2. If down -> try to start it.
#   3. If it still will not come up -> STOP the scheduler (dispatch worker),
#      because the operator's rule is: no rerouter => the scheduler must not run.
#   4. When the rerouter is healthy again -> the dispatch worker's own
#      while-true shell brings it back, so recovery needs no action here.
#
# Randomised, per the operator's "随机检测": a fixed period is predictable and
# can beat against the 120s reroute cycle; a jittered one samples all phases.
set -u
RR_SCRIPT="$HOME/work/tpu_cmd/tpu_reroute_loop_v17.sh"
RR_LOCK=/tmp/tpu-reroute-loop.lock
LOG="$HOME/work/.monitor_watch/tpu_interlock_v17.log"
DISPATCH_SHELL_PID=2809021   # the while-true wrapper that respawns the worker

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

rr_alive() {
  # The lock is the single source of truth for "an instance is running":
  # flock -n succeeds only if NOBODY holds it.
  if flock -n "$RR_LOCK" -c true 2>/dev/null; then return 1; else return 0; fi
}

exec 8>/tmp/tpu-interlock.lock
flock -n 8 || { echo "interlock already running"; exit 0; }

while true; do
  if rr_alive; then
    log "ok: rerouter holds the lock"
  else
    log "ALERT: rerouter DOWN -> auto-starting"
    setsid nohup bash "$RR_SCRIPT" >/dev/null 2>&1 &
    sleep 20
    if rr_alive; then
      log "recovered: rerouter restarted"
    else
      log "FATAL: rerouter will not start -> halting scheduler (SIGSTOP dispatch $DISPATCH_SHELL_PID)"
      kill -STOP "$DISPATCH_SHELL_PID" 2>/dev/null \
        && log "scheduler HALTED (SIGSTOP, reversible with kill -CONT)" \
        || log "could not signal dispatch shell $DISPATCH_SHELL_PID (gone?)"
    fi
  fi
  SLEEP=$(( 90 + RANDOM % 120 ))   # 90-210s, jittered
  sleep "$SLEEP"
done
