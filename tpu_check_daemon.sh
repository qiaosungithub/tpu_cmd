#!/bin/bash
# Which operator's board this daemon maintains. Unset = sqa's own files, which
# is exactly how this behaved before the variables existed. A second operator
# on the same Unix account (lyy, via the `npu` function in tpu_wrapper.sh)
# needs a SECOND daemon process started with these pointed at their files —
# `npu check` renders from $TPU_CHECK_CACHE_FILE, and with nobody writing it
# every job on that board reads SUBMITTED forever.
#
#   TPU_JOBS_FILE=~/lyy-work/.npu_jobs.json \
#   TPU_CHECK_CACHE_FILE=~/lyy-work/.npu_check_cache.txt \
#   TPU_CHECK_TIME_FILE=~/lyy-work/.npu_check_time.txt \
#     bash tpu_check_daemon.sh
: "${TPU_JOBS_FILE:=$HOME/.tpu_jobs.json}"
: "${TPU_CHECK_CACHE_FILE:=$HOME/.tpu_check_cache.txt}"
: "${TPU_CHECK_TIME_FILE:=$HOME/.tpu_check_time.txt}"
export TPU_JOBS_FILE TPU_CHECK_CACHE_FILE
CACHE_FILE="$TPU_CHECK_CACHE_FILE"
TMP_FILE="$TPU_CHECK_CACHE_FILE.tmp"
TIME_FILE="$TPU_CHECK_TIME_FILE"

# SMART-ROUTER LANE (4th lane), OFF BY DEFAULT. When TPU_ROUTE_ENABLED=1 the
# daemon drains the local queue (~/.tpu_local_queue.json) into the XM queue on
# each round and re-routes jobs stuck PENDING past the deadline. Left unset it
# is a complete no-op -- the daemon behaves exactly as before. TPU_ROUTE_DRYRUN=1
# (default) makes the lane plan-and-log only; set TPU_ROUTE_DRYRUN=0 to actually
# submit and cancel. The queue file is operator-scoped like the registry.
: "${TPU_ROUTE_ENABLED:=0}"
: "${TPU_ROUTE_DRYRUN:=1}"
: "${TPU_LOCAL_QUEUE_FILE:=$HOME/.tpu_local_queue.json}"
: "${TPU_ROUTE_GROUP:=9}"
: "${TPU_ROUTE_REROUTE_AFTER_S:=600}"
export TPU_LOCAL_QUEUE_FILE

cd /google/src/cloud/qiaos/xm_test/google3 || {
    echo "Directory not found!"
    sleep 60
    exit 1
}

echo "tpu_check daemon started. Refreshing every ~20s after each round..."

# Where the checker binaries live.
#
# `blaze-bin` is a symlink CHAIN whose two hops have OPPOSITE lifetimes:
#
#   blaze-bin -> $output_base/execroot/google3/blaze-out/k8-fastbuild/bin  (stable)
#   blaze-out -> /google/obj/workspace/namespace/<uuid>/blaze-out         (NEW PER BUILD)
#
# The first hop is repointed only when the build configuration changes, so
# pinning it protects a running round from a concurrent build in another
# terminal. The second hop is republished by EVERY build, and only that
# build's targets land in the new namespace.
#
# `readlink -f` collapsed both hops and froze the daemon inside one build's
# objfs namespace. A later single-target build then published a namespace
# without money_check/quota_check while the daemon kept looking at the old
# one, and no rebuild could ever reach it: the binaries sat in blaze-bin,
# perfectly built, for 17 hours of "checker not built" and stale prices.
#
# So: resolve ONE hop at startup, and follow the objfs hop at every use.
G3="/google/src/cloud/qiaos/xm_test/google3"
CHECKER_SUBDIR="experimental/users/qiaos/tpu_utils"
CHECKER_NAMES="money_check quota_check infra_check"
BLAZE_BIN="$(readlink ./blaze-bin 2>/dev/null || echo "$G3/blaze-bin")"
CHECKER_DIR="$BLAZE_BIN/$CHECKER_SUBDIR"
BUILD_HINT="(cd $G3 && blaze build $CHECKER_SUBDIR:{money_check,quota_check,infra_check})"
BUILD_HINT_ROUTE="(cd $G3 && blaze build $CHECKER_SUBDIR:route_check)"

# Usable path for checker $1, or nothing. Resolved at every use, and falling
# back to the live symlink so a config change cannot strand the daemon.
checkerbin() {
  local name="$1" c
  for c in "$CHECKER_DIR/$name" "$G3/blaze-bin/$CHECKER_SUBDIR/$name"; do
    if [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

# --- self-heal -----------------------------------------------------------
# A missing checker used to be logged forever and fixed by nobody: the log
# said "build it", the log was in a detached tmux pane nobody reads, and
# `tpu money`'s auto-recovery restarted the session instead, which cannot
# rebuild anything. Repair it here, where the failure is detected.
#
# Build ALL THREE in ONE invocation. Building one target publishes an output
# namespace containing only that target, which is precisely the failure being
# repaired -- a one-target repair would re-create the bug.
REBUILD_LOG="/tmp/tpu_daemon_selfheal_build.log"
REBUILD_MIN_INTERVAL=1800   # s; a rebuild costs ~70s even fully cached
LAST_REBUILD=0

self_heal_checkers() {
  local missing="" name now
  for name in $CHECKER_NAMES; do
    checkerbin "$name" >/dev/null || missing="$missing $name"
  done
  [ -z "$missing" ] && return 0

  now=$(date +%s)
  if [ $(( now - LAST_REBUILD )) -lt "$REBUILD_MIN_INTERVAL" ]; then
    echo "$(date): checkers missing ($missing ) but a self-heal build ran $(( now - LAST_REBUILD ))s ago; not rebuilding."
    return 1
  fi
  LAST_REBUILD=$now
  echo "$(date): SELF-HEAL - missing:$missing -> rebuilding all of them ($REBUILD_LOG)"
  if ( cd "$G3" && blaze build \
         "$CHECKER_SUBDIR:money_check" \
         "$CHECKER_SUBDIR:quota_check" \
         "$CHECKER_SUBDIR:infra_check" ) > "$REBUILD_LOG" 2>&1; then
    # The rebuild republishes blaze-out; re-resolve the stable hop too, in
    # case this build also moved blaze-bin itself.
    BLAZE_BIN="$(cd "$G3" && readlink ./blaze-bin 2>/dev/null || echo "$G3/blaze-bin")"
    CHECKER_DIR="$BLAZE_BIN/$CHECKER_SUBDIR"
    echo "$(date): SELF-HEAL - rebuild OK"
  else
    echo "$(date): SELF-HEAL - rebuild FAILED, see $REBUILD_LOG. Last lines:"
    tail -3 "$REBUILD_LOG" | sed 's/^/    | /'
  fi
}

# --- checker tasks -------------------------------------------------------
# The three checkers are independent: they share no state and write to
# disjoint outputs (infra -> $CACHE_FILE, quota -> ~/.tpu_quota_cache_dir/*,
# money -> ~/.tpu_quota_cache_dir/money.txt). Each pays a ~15s Python/par
# cold-start tax, while its actual RPCs cost <1s. Run serially that tax was
# paid three times and a round took ~216s, which exceeded the 180s freshness
# threshold in tpu_wrapper.sh and made the staleness alarm fire every cycle.
# Running them concurrently collapses the round to roughly the slowest task.

run_infra_check() {
  local bin
  bin=$(checkerbin infra_check) || {
    echo "$(date): FATAL - infra checker not built or not executable."
    echo "$(date):         Build it:  $BUILD_HINT"
    return 1
  }
  "$bin" 2>/dev/null | grep -Ev "274311238|274310306|274303586|274276782|274276523|274275526|274274881|274274856|274269375|274256617|274256088|274255958|274456072|274454581" > "$TMP_FILE"
  if [ -s "$TMP_FILE" ]; then
    mv "$TMP_FILE" "$CACHE_FILE"
    date +%s > "$TIME_FILE"
    echo "$(date): Successfully updated infra cache"
  else
    echo "$(date): Error - infra output is empty. Retrying next round..."
  fi
}


# SMART-ROUTER LANE. Off unless TPU_ROUTE_ENABLED=1. Drains the local queue
# (place pass) then re-routes jobs stuck PENDING (reroute pass). Each pass is
# its own route_check invocation reading/writing the same operator-scoped queue
# file. --nodry_run only when TPU_ROUTE_DRYRUN=0; otherwise it plans and logs.
# Like infra, this makes serial XManager/goodput RPCs, so it runs DETACHED off
# the fast lane -- money/quota must never wait on it.
run_route_lane() {
  local bin dry_flag
  bin=$(checkerbin route_check) || {
    echo "$(date): route lane - route_check not built; skipping (build: $BUILD_HINT_ROUTE)"
    return 1
  }
  if [ "$TPU_ROUTE_DRYRUN" = "0" ]; then dry_flag="--nodry_run"; else dry_flag="--dry_run"; fi
  echo "$(date): route lane - place pass ($dry_flag, queue=$TPU_LOCAL_QUEUE_FILE)"
  "$bin" --queue_file="$TPU_LOCAL_QUEUE_FILE" --group="$TPU_ROUTE_GROUP" "$dry_flag" 2>&1 \
    | sed 's/^/  [route:place] /'
  echo "$(date): route lane - reroute pass ($dry_flag, after ${TPU_ROUTE_REROUTE_AFTER_S}s)"
  "$bin" --queue_file="$TPU_LOCAL_QUEUE_FILE" --reroute \
    --reroute_after_s="$TPU_ROUTE_REROUTE_AFTER_S" "$dry_flag" 2>&1 \
    | sed 's/^/  [route:reroute] /'
  echo "$(date): route lane - done"
}


# $1 = binary name, $2 = human-readable label for logs.
run_named_check() {
  local bin out rc

  # A MISSING BINARY IS NOT A TRANSIENT FAILURE. This used to fall through to
  # the generic "Retrying next round...", so a checker that could never succeed
  # looked exactly like one waiting out a blip: the cache silently froze while
  # the log scrolled reassuring retry messages, and `tpu money`'s auto-recovery
  # dutifully restarted a tmux session that was never the problem. Say the one
  # thing that fixes it instead.
  bin=$(checkerbin "$1") || {
    echo "$(date): FATAL - $2 checker not built or not executable: $CHECKER_DIR/$1"
    echo "$(date):         Build it:  $BUILD_HINT"
    return 1
  }

  # Capture status from the BINARY. `local out=$(...)` would set $? from the
  # `local` builtin, not from the command -- the old code declared `local out`
  # separately for exactly this reason, so keep the two statements apart.
  out=$("$bin" 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "$(date): Successfully updated $2 cache directory"
    return 0
  fi

  if echo "$out" | grep -qiE "unauthenticated|loas|gcert|permission|denied|auth"; then
    echo "$(date): Error - $2 check failed due to gcert/LOAS authentication! Please run 'gcert'."
  else
    # Show the tail. A bare "check failed" is unactionable, and the reason is
    # already in hand.
    echo "$(date): Error - $2 check failed (rc=$rc). Retrying next round. Last output:"
    echo "$out" | tail -3 | sed 's/^/    | /'
  fi
  return $rc
}

# The slow-lane infra pass runs detached; reap it on exit so the outer
# `while true; do tpu_check_daemon.sh; sleep 5; done` wrapper cannot accumulate
# orphaned infra children across daemon restarts.
trap '[ -n "${INFRA_PID:-}" ] && kill "$INFRA_PID" 2>/dev/null; [ -n "${ROUTE_PID:-}" ] && kill "$ROUTE_PID" 2>/dev/null' EXIT

while true; do
  ROUND_START=$(date +%s)

  # Repair a missing binary before the round rather than logging about it.
  self_heal_checkers

  # Each task logs to its own temp file so concurrent writes cannot interleave;
  # the logs are replayed in a stable order once the round completes.
  # FAST LANE: money + quota. They cost ~40s and the round WAITS for them, so
  # neither cache ever ages past the 300s staleness alarm in tpu_wrapper.sh.
  # They used to share a single `wait` barrier with infra_check below, which on
  # a large registry takes minutes -- so money.txt aged to ~386s and `tpu money`
  # cried stale even though its own data was ready in 37s.
  LOG_QUOTA=$(mktemp)
  LOG_MONEY=$(mktemp)
  run_named_check quota_check quota > "$LOG_QUOTA" 2>&1 &
  PID_QUOTA=$!
  run_named_check money_check money > "$LOG_MONEY" 2>&1 &
  PID_MONEY=$!
  wait "$PID_QUOTA" "$PID_MONEY"
  cat "$LOG_QUOTA" "$LOG_MONEY"
  rm -f "$LOG_QUOTA" "$LOG_MONEY"

  # SLOW LANE: infra_check issues one serial XManager RPC per tracked
  # experiment, so it scales with registry size and takes minutes. It is now
  # DETACHED -- the round never waits on it, so a slow infra pass can no longer
  # starve the fast lane. A kill -0 guard skips starting a second pass while one
  # is still in flight, so passes queue instead of stacking up.
  if [ -z "${INFRA_PID:-}" ] || ! kill -0 "$INFRA_PID" 2>/dev/null; then
    run_infra_check &
    INFRA_PID=$!
    INFRA_START=$(date +%s)
  else
    echo "$(date): infra pass still running ($(( $(date +%s) - ${INFRA_START:-0} ))s); skipping this round"
  fi

  # ROUTER LANE (4th lane), OFF unless TPU_ROUTE_ENABLED=1. Same discipline as
  # infra: DETACHED so its serial RPCs never delay money/quota, with a kill -0
  # guard so a slow pass does not stack. A no-op when disabled.
  if [ "$TPU_ROUTE_ENABLED" = "1" ]; then
    if [ -z "${ROUTE_PID:-}" ] || ! kill -0 "$ROUTE_PID" 2>/dev/null; then
      run_route_lane &
      ROUTE_PID=$!
      ROUTE_START=$(date +%s)
    else
      echo "$(date): route lane still running ($(( $(date +%s) - ${ROUTE_START:-0} ))s); skipping this round"
    fi
  fi

  echo "$(date): --- fast-lane round took $(( $(date +%s) - ROUND_START ))s ---"
  
  
  # Parse xm launch logs for ERROR MODES
  python3 - << 'EOF'
import json, os, fcntl, time, re, sys, subprocess, traceback

mapping_file = os.environ.get('TPU_JOBS_FILE') or os.path.expanduser('~/.tpu_jobs.json')
if os.path.exists(mapping_file):
    try:
        with open(mapping_file, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        
        changed = False

        # Parse cache for why
        cache_file = os.environ.get('TPU_CHECK_CACHE_FILE') or os.path.expanduser('~/.tpu_check_cache.txt')
        cached_status = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    for line in f:
                        line = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line).strip()
                        if line.startswith('│') and line.endswith('│'):
                            parts = [p.strip() for p in line.split('│')[1:-1]]
                            if len(parts) >= 3 and parts[0].isdigit():
                                cached_status[parts[0]] = {'status': parts[1], 'why': parts[-1] if len(parts) >= 4 else ''}
            except Exception:
                pass

        for xid, info in data.items():
            if info.get('status') not in ['FAILED', 'DONE']:
                log_file = info.get('launch_log', '')
                if log_file and os.path.exists(log_file):
                    with open(log_file, 'r') as log_f:
                        content = log_f.read()
                    
                    if 'SLICE_DEFRAGMENTATION' in content:
                        info['status'] = 'FAILED'
                        info['error'] = 'Preempted by Defag'
                        changed = True
                    elif 'RESOURCE_EXHAUSTED' in content or 'RESOURCES_EXCEEDED' in content:
                        info['status'] = 'FAILED'
                        info['error'] = 'Resource Exhausted (Topology/Quota)'
                        changed = True
                    elif 'FAILED' in content and 'Preempted' not in content:
                        info['status'] = 'FAILED'
                        info['error'] = 'Unknown failure in XManager logs'
                        changed = True
                    elif 'All work units started' in content:
                        if info['status'] != 'RUNNING':
                            info['status'] = 'RUNNING'
                            changed = True

        cmds_to_run = []
        now = time.time()
        for xid, info in list(data.items()):
            c_info = cached_status.get(xid, {})
            c_status = c_info.get('status', '').lower()
            if 'failed' in c_status and info.get('status') != 'FAILED':
                info['status'] = 'FAILED'
                changed = True
                
            why = c_info.get('why') or info.get('error') or ''
            if not why or why == 'unknown reason':
                if 'failed' in c_status or info.get('status') == 'FAILED':
                    why = 'Rejected by Allocator/Borg'
                    
            status = info.get('status')
            retry_count = info.get('retry_count', 5)
            tier = info.get('tier', '-')
            
            if status == 'FAILED' and tier == 'PROD' and 'Rejected by Allocator/Borg' in why:
                if retry_count < 5:
                    last_time = info.get('retry_timer_start', now)
                    if 'retry_timer_start' not in info:
                        info['retry_timer_start'] = now
                        changed = True
                        last_time = now
                        
                    if now - last_time >= 300:
                        info['retry_count'] = retry_count + 1
                        info['retry_timer_start'] = now
                        changed = True
                        
                        stagedir_abs = info.get('stagedir')
                        if stagedir_abs and not stagedir_abs.startswith('/'):
                            stagedir_abs = f'/google/src/cloud/qiaos/EqR-jax/google3/{stagedir_abs}'
                            
                        launch_log = info.get('launch_log', '')
                        cmd = None
                        if os.path.exists(launch_log):
                            with open(launch_log, 'r') as log_f:
                                for line in log_f:
                                    if line.startswith('Running:'):
                                        cmd = line[len('Running:'):].strip()
                                        break
                                        
                        if cmd and stagedir_abs:
                            cmds_to_run.append((xid, f'cd "{stagedir_abs}" && {cmd} 2>&1 | tee "{launch_log}"'))
                            
        if changed:
            with open(mapping_file, 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(data, f, indent=2)
                fcntl.flock(f, fcntl.LOCK_UN)
            print(f"Successfully updated {mapping_file}")
            
        for xid, cmd in cmds_to_run:
            print(f"RETRYING JOB {xid}: {cmd}")
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            log_c = p.stdout
            m = re.search(r"Launched experiment (\d+)", log_c)
            if m:
                new_xid = m.group(1)
                with open(mapping_file, 'r') as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    d = json.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
                
                if xid in d:
                    old_info = d.pop(xid)
                    d[new_xid] = old_info
                    d[new_xid]['status'] = 'SUBMITTED'
                    d[new_xid]['error'] = ''
                    
                    with open(mapping_file, 'w') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        json.dump(d, f, indent=2)
                        fcntl.flock(f, fcntl.LOCK_UN)
                    print(f'{time.ctime()}: Replaced old XID {xid} with new XID {new_xid}')
    except Exception as e:
        print(f'{time.ctime()}: Error parsing logs: {e}')
        traceback.print_exc()
EOF

  # A round now costs ~60s wall time instead of ~216s. Sleeping 20s keeps the
  # worst-case cache age well under the 180s threshold in tpu_wrapper.sh.
  sleep 20
done
