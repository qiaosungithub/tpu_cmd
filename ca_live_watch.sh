#!/bin/bash
# Watch for the FIRST real content-addressed build to go through the live queue,
# and notify the chatbot session when it reaches an outcome. Not a busy-poll on
# the agent side: this runs in the background and pings once on a real event.
#
# Success  = a NEW registry xid whose stagedir basename is eqr_run_ca_* reaches
#            SUBMITTED or RUNNING (XManager accepted the CA target on a live job).
# Failure  = such an xid reaches FAILED, OR a new provenance line has fallback=1
#            (correctness preserved, but the cache was missed for a reason worth
#            surfacing), OR a [[STAGE_*]] marker shows up in a fresh xm_launch log.
# Timeout  = after MAX_MIN minutes, ping so the watch never dies silently.
set -u

SESSION="${1:-chatty-bot}"
NOTIFY="$HOME/.amply/bin/amply_notify"
JOBS="$HOME/.tpu_jobs.json"
PROV="$HOME/.tpu_ca_provenance.jsonl"
PARENT=/google/src/cloud/qiaos/run_amply_workspace/google3/experimental/qiaos/eqr_jax_final_stages
POLL="${CA_WATCH_POLL:-30}"
MAX_MIN="${CA_WATCH_MAX_MIN:-480}"
LOG=/tmp/ca_live_watch.log

say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
ping() { "$NOTIFY" "$SESSION" "$1" >>"$LOG" 2>&1; say "NOTIFY rc=$? : $1"; }

say "=== ca_live_watch start (session=$SESSION poll=${POLL}s max=${MAX_MIN}m) ==="

# Sanity: the durable enable sentinel must be present, else the worker is on the
# default path and no eqr_run_ca_* will ever appear -- warn once so a silent
# watch is not mistaken for "all healthy, just no traffic".
SENTINEL="${JOBS%/*}/.tpu_local_queue.ca_enabled"
[ -f "$HOME/.tpu_local_queue.ca_enabled" ] || ping "⚠️ 提醒: 未发现 CA 启用 sentinel (~/.tpu_local_queue.ca_enabled);当前 tpu worker 可能在默认暂存路径,不会产生 eqr_run_ca_* 。请回来确认。"

# Baseline: xids already in the registry, and current provenance line count, so
# we only react to NEW activity.
baseline_xids=$(python3 - "$JOBS" <<'PY'
import json,os,sys
try: j=json.load(open(os.path.expanduser(sys.argv[1])))
except Exception: j={}
print(" ".join(j.keys()) if isinstance(j,dict) else "")
PY
)
prov_base=0
[ -f "$PROV" ] && prov_base=$(wc -l < "$PROV")
say "baseline xids=$(echo $baseline_xids | wc -w) prov_lines=$prov_base"

deadline=$(( $(date +%s) + MAX_MIN*60 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  # 1. New registry xid with a content-addressed stagedir reaching an outcome.
  hit=$(python3 - "$JOBS" "$baseline_xids" <<'PY'
import json,os,sys
jobs=os.path.expanduser(sys.argv[1]); base=set(sys.argv[2].split())
try: j=json.load(open(jobs))
except Exception: j={}
for xid,r in (j.items() if isinstance(j,dict) else []):
    if xid in base or not isinstance(r,dict): continue
    sd=str(r.get('stagedir','')); bn=os.path.basename(sd.rstrip('/'))
    if not bn.startswith('eqr_run_ca_'): continue
    st=str(r.get('status',''))
    if st in ('SUBMITTED','RUNNING'):
        print(f"SUCCESS\t{xid}\t{st}\t{bn}\t{r.get('exp_name','')}"); break
    if st in ('FAILED','CANCELLED'):
        print(f"FAIL\t{xid}\t{st}\t{bn}\t{str(r.get('error',''))[:160]}"); break
PY
)
  if [ -n "$hit" ]; then
    kind=$(echo "$hit" | cut -f1); xid=$(echo "$hit" | cut -f2); st=$(echo "$hit" | cut -f3)
    bn=$(echo "$hit" | cut -f4); extra=$(echo "$hit" | cut -f5)
    if [ "$kind" = "SUCCESS" ]; then
      ping "✅ 线上首个内容寻址构建绿了: XID $xid 状态=$st stagedir=$bn exp=$extra 。请回来确认收尾。"
    else
      ping "⚠️ 线上内容寻址 job XID $xid 状态=$st stagedir=$bn 详情=$extra 。请回来查看。"
    fi
    say "outcome delivered ($kind $xid); exiting"; exit 0
  fi

  # 2. New provenance line with fallback=1 (correctness kept, cache missed).
  if [ -f "$PROV" ]; then
    now_lines=$(wc -l < "$PROV")
    if [ "$now_lines" -gt "$prov_base" ]; then
      newfb=$(tail -n +"$((prov_base+1))" "$PROV" | grep -c '"fallback": true' 2>/dev/null || echo 0)
      newpub=$(( now_lines - prov_base ))
      say "provenance +$newpub line(s), fallback=$newfb"
      if [ "${newfb:-0}" -gt 0 ]; then
        ping "⚠️ 内容寻址发布走了 fallback 分支(正确性无损,但未命中缓存): $PROV 最新行 fallback=true。请回来查看原因。"
        say "fallback notified; exiting"; exit 0
      fi
      prov_base=$now_lines
    fi
  fi
  sleep "$POLL"
done

ping "⏳ 内容寻址监视器已运行 ${MAX_MIN} 分钟仍未见线上真实 CA 构建(队列可能一直空)。监视器已退出,需要的话我再起一个。"
say "timeout; exiting"
