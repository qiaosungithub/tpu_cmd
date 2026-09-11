#!/bin/bash
# infra-v18: watch for the FIRST real car dispatched AFTER the single-claimer fix
# (old --worker stopped 04:29Z). Sample the group field every 15s for >=240s.
# Criterion (FLEET_STANDING 3ao): group non-None AND survives past 210s.
Q="$HOME/.tpu_local_queue.json"
CUT=${CUT:-1788150540}   # 2026-08-31 04:29:00Z, when --worker was stopped
OUT="$HOME/work/.monitor_watch/v18_realcar_verify.log"
NOTIFY="$HOME/.amply/bin/amply_notify"
SESSION="${1:?need session id}"
seen=""
while true; do
  cand=$(python3 - "$Q" "$CUT" <<'PY'
import json,sys,os
q,cut=sys.argv[1],float(sys.argv[2])
try: d=json.load(open(q))
except Exception: sys.exit(0)
rows=d if isinstance(d,list) else list(d.values())[0]
for x in rows:
    if not isinstance(x,dict): continue
    st=x.get('state')
    if st in ('QUEUED','BUILD_REQUESTED','BUILDING','SUBMITTED','RUNNING'):
        try: sub=float(x.get('submitted_at') or 0)
        except Exception: sub=0
        if sub>cut: print(x.get('job_id'), st, sub)
PY
)
  while read -r jid st sub; do
    [ -z "$jid" ] && continue
    case " $seen " in *" $jid "*) continue;; esac
    seen="$seen $jid"
    echo "[$(date -u +%H:%M:%SZ)] NEW REAL CAR $jid state=$st sub=$sub -- starting 240s group sampling" >>"$OUT"
    "$NOTIFY" "$SESSION" "REAL CAR DETECTED post-fix: $jid (state=$st). Starting 240s group sampling now; will report at t=210s/240s." 2>/dev/null
    (
      for t in $(seq 15 15 240); do
        sleep 15
        line=$(python3 - "$Q" "$jid" <<'PY'
import json,sys
q,jid=sys.argv[1],sys.argv[2]
try: d=json.load(open(q))
except Exception: print("READ_FAIL"); raise SystemExit
rows=d if isinstance(d,list) else list(d.values())[0]
for x in rows:
    if isinstance(x,dict) and x.get('job_id')==jid:
        print(f"state={x.get('state')} group={x.get('group')} xid={x.get('xid')} att={x.get('attempts')} rer={x.get('reroutes')}")
        raise SystemExit
print("GONE_FROM_QUEUE")
PY
)
        echo "  t=${t}s $line" >>"$OUT"
      done
      final=$(tail -1 "$OUT")
      "$NOTIFY" "$SESSION" "240s sampling DONE for $jid. Last: $final . Full log: $OUT" 2>/dev/null
    ) &
  done <<< "$cand"
  sleep 10
done
