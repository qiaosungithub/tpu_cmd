#!/bin/bash
# Restart ALL THREE queue writers once BUILDING drains (infra-v17).
# ★Enumerate by fd/lock, never from memory: two earlier rounds each missed the
# rerouter because --reroute_loop is not adjacent to the binary name.
# ★Kill only the ENGINE pids, not their while-true shells (the shells respawn).
LOG="$HOME/work/.monitor_watch/i17_restart_groupdup.log"
exec 7>/tmp/i17-restart-groupdup.lock; flock -n 7 || exit 0
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
source "$HOME/work/tpu_cmd/tpu_wrapper.sh" >/dev/null 2>&1
for i in $(seq 1 60); do
  out=$(tpu queue-status 2>&1)
  n=$(printf '%s' "$out" | sed -r 's/\x1B\[[0-9;]*[mK]//g' | awk '$2=="BUILDING"||$2=="BUILD_REQUESTED"' | wc -l)
  if [ "$n" -eq 0 ]; then
    say "BUILDING=0 -> restarting all three writers"
    for p in $(ps -eo pid,args | awk '/route_check --worker/ && !/npu/ && !/bash -c/ && !/awk/{print $1}'); do
      kill -9 "$p" 2>/dev/null && say "killed build-worker $p"; done
    for p in $(ps -eo pid,args | awk '/route_check/ && /dispatch_worker/ && !/npu/ && !/bash -c/ && !/awk/{print $1}'); do
      kill -9 "$p" 2>/dev/null && say "killed dispatch $p"; done
    for p in $(ls /proc|grep -E '^[0-9]+$'); do
      ls -l /proc/$p/fd 2>/dev/null | grep -q 'tpu-reroute-loop.lock' && { kill -9 $p 2>/dev/null; say "killed rerouter holder $p"; }
    done
    sleep 30
    say "after: $(ps -eo pid,args | awk '/route_check --(worker|dispatch_worker)/ && !/npu/ && !/bash -c/ && !/awk/{printf \"%s \", $1}')"
    exit 0
  fi
  say "waiting: BUILDING=$n (tick $i) -- codi eval 066f03, read-only"
  sleep 60
done
say "TIMEOUT; writers NOT restarted"
