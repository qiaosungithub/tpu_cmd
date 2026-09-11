#!/bin/bash
# Restart the two TPU queue writers ONCE BUILDING drains, so they pick up the
# new route_lib (reroutes counter). v16's rule: a fix only affects NEW
# processes -- and you must not kill a worker mid-build (that is elt's car).
# NPU writer 2474741 is deliberately untouched (red line: audit only).
LOG="$HOME/work/.monitor_watch/i17_restart_writers.log"
exec 7>/tmp/i17-restart-writers.lock; flock -n 7 || exit 0
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
source "$HOME/work/tpu_cmd/tpu_wrapper.sh" >/dev/null 2>&1
for i in $(seq 1 60); do
  out=$(tpu queue-status 2>&1)
  n=$(printf '%s' "$out" | sed -r 's/\x1B\[[0-9;]*[mK]//g' \
        | awk '$2=="BUILDING"||$2=="BUILD_REQUESTED"' | wc -l)
  if [ "$n" -eq 0 ]; then
    say "BUILDING=0 -> restarting writers"
    for p in 1850152 2809023; do
      pp=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
      kill -9 "$p" 2>/dev/null && say "killed worker $p (its while-true shell $pp respawns it)"
    done
    sleep 25
    say "after restart: $(ps -eo pid,args | awk '/route_check --(worker|dispatch_worker)/ && !/npu/ && !/awk/{printf "%s ", $1}')"
    exit 0
  fi
  say "waiting: BUILDING/BUILD_REQUESTED=$n (tick $i)"
  sleep 60
done
say "TIMEOUT after 60 ticks; writers NOT restarted"
