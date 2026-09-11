#!/bin/bash
# infra-v18: report the FIRST real completed->DONE produced by the new reconcile.
# Two independent instruments, because a counter that never moves is
# indistinguishable from a watcher pointed at the wrong file:
#   A) the rerouter's own log line  `N completed->DONE` with N>0
#   B) the queue itself growing a row whose state == DONE
LOG=~/work/.monitor_watch/tpu_reroute_loop_v17.log
Q="$HOME/.tpu_local_queue.json"
OUT=~/work/.monitor_watch/v18_done_first.log
NOTIFY="$HOME/.amply/bin/amply_notify"
SESSION="${1:?need session id}"
while true; do
  hit=$(grep -a 'completed->DONE' "$LOG" | grep -avE '(^|[^0-9])0 completed->DONE' | tail -1)
  rows=$(python3 - "$Q" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: raise SystemExit
rows=d if isinstance(d,list) else list(d.values())[0]
out=[f"{x.get('job_id')}(xid={x.get('xid')})" for x in rows
     if isinstance(x,dict) and x.get('state')=='DONE']
print(' '.join(out))
PY
)
  if [ -n "$hit" ] || [ -n "$rows" ]; then
    {
      echo "=== FIRST REAL completed->DONE @ $(date -u +%FT%TZ) ==="
      echo "log line : ${hit:-<none>}"
      echo "DONE rows: ${rows:-<none>}"
    } >>"$OUT"
    "$NOTIFY" "$SESSION" "FIRST completed->DONE FIRED. log='${hit:-none}' DONE_rows='${rows:-none}'" 2>/dev/null
    exit 0
  fi
  sleep 20
done
