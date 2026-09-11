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
# ============================================================================
# ★TPU LANE DISABLED -- scheduler rewrite in progress (infra-v12, 2026-08-28)
# ============================================================================
# The TPU scheduler was replaced today (one job, one chain; version-CAS writes;
# the queue now lives in ~/.tpu_jobs_v2.json). This daemon still reads the OLD
# queue ~/.tpu_local_queue.json, so if it runs alongside the new dispatcher the
# two of them drain two different files -- which is how one config produced
# five concurrent 8-card jobs earlier today.
#
# It was stopped three times and restarted three times (22:22Z it got as far as
# dispatching a real job). Chasing the restarter through the process tree does
# not work: every candidate is a `bash -c` under the shared tmux server, and a
# setsid'd child loses the link. So the fix is the same one applied to the
# poisoned bucket: REFUSE, rather than guess who will make the mistake.
#
# ★NPU lane is unaffected and must stay that way -- lyy's board depends on it.
# The discriminator is TPU_LOCAL_QUEUE_FILE, which the NPU launcher always sets
# and the TPU side never does. NOT the script path: ~/work/tpu_check_daemon.sh
# is a symlink to this same file, so both lanes run identical bytes.
#
# TO RE-ENABLE: delete this block (infra-v12 will, at the end of the rewrite),
# or export TPU_DAEMON_FORCE=1 for a one-off deliberate run.
case "${TPU_LOCAL_QUEUE_FILE:-}" in
  *npu_local_queue*) : ;;                      # NPU lane: carry on
  *)
    # ★READ-ONLY LANES STAY ON; ONLY THE WRITERS ARE GATED (infra-v16, 2026-08-30).
    #
    # infra-v12 blocked this daemon with a bare `exit 0` while the scheduler was
    # rewritten. That was too wide: the thing it needed to stop was the DISPATCH
    # pass (a second drainer on the old queue), but `exit 0` also killed the
    # infra/quota/money passes, which only READ and then refresh caches.
    #
    # Cost of the over-block, measured 2026-08-30: `~/.tpu_check_cache.txt` froze
    # for 41h, so `tpu check` rendered 66 "active" jobs that were mostly dead
    # (284831213 showed SUBMITTED for 12h27m while XM had it NOT_RUNNING), and
    # budget_enforcer -- which prices the fleet from that same cache -- built kill
    # lists out of corpses and "reclaimed" credits that were never being spent.
    # A stale cache is not a display bug; anything that budgets off it is wrong.
    #
    # So: run the read-only passes, and pin the two writer passes OFF. The
    # standalone `route_check --dispatch_worker` (tpu-build-worker) remains the
    # SOLE drainer, which is exactly what TPU_ROUTE_INLANE_PLACE=0 already means.
    # An explicit env value still wins, so a deliberate override is unchanged.
    if [ "${TPU_DAEMON_FORCE:-0}" != "1" ]; then
      : "${TPU_ROUTE_INLANE_PLACE:=0}"
      : "${TPU_ROUTE_INLANE_REROUTE:=0}"
      export TPU_ROUTE_INLANE_PLACE TPU_ROUTE_INLANE_REROUTE
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [tpu_check_daemon] TPU lane: read-only passes" \
           "ON (infra/quota/money refresh the caches that \`tpu check\` and" \
           "budget_enforcer read); dispatch+reroute passes PINNED OFF --" \
           "tpu-build-worker is the sole drainer of the local queue." >&2
    fi
    ;;
esac

: "${TPU_JOBS_FILE:=$HOME/.tpu_jobs.json}"
: "${TPU_CHECK_CACHE_FILE:=$HOME/.tpu_check_cache.txt}"
: "${TPU_CHECK_TIME_FILE:=$HOME/.tpu_check_time.txt}"
export TPU_JOBS_FILE TPU_CHECK_CACHE_FILE
CACHE_FILE="$TPU_CHECK_CACHE_FILE"
TMP_FILE="$TPU_CHECK_CACHE_FILE.tmp"
TIME_FILE="$TPU_CHECK_TIME_FILE"

# SMART-ROUTER LANE (4th lane), ON BY DEFAULT (operator decision 2026-08-24).
# The daemon drains the local queue (~/.tpu_local_queue.json) into the XM queue
# on each round (place pass) and re-routes jobs stuck PENDING past the deadline
# (reroute pass): a SUBMITTED job still PENDING > TPU_ROUTE_REROUTE_AFTER_S is
# cancelled and returned to QUEUED, with its stuck cell cooled so the next plan
# avoids it. Set TPU_ROUTE_ENABLED=0 to disable the whole lane. TPU_ROUTE_DRYRUN
# now defaults to 0 (live: actually submit and cancel); set TPU_ROUTE_DRYRUN=1
# to fall back to plan-and-log only. The queue file is operator-scoped.
: "${TPU_ROUTE_ENABLED:=1}"
: "${TPU_ROUTE_DRYRUN:=0}"
: "${TPU_LOCAL_QUEUE_FILE:=$HOME/.tpu_local_queue.json}"
: "${TPU_ROUTE_GROUP:=9}"
# Placement group PREFERENCE ORDER (operator 2026-08-24: lean on the free vqfree
# pool before the paid g9 floor). The place pass tries these groups in order and
# only lets what an earlier group cannot place fall through to the next. Default
# "5,9": vqfree first, g9 floor as fallback. Set to a single group (e.g. "9") to
# restore the old single-group behaviour.
: "${TPU_ROUTE_GROUP_ORDER:=5,9}"
: "${TPU_ROUTE_REROUTE_AFTER_S:=600}"
# in-lane reroute pass (Step2 Phase B): default 1 = daemon runs its own reroute
# pass in the route lane (historical behavior). Set to 0 to SKIP it -- used when a
# standalone `route_check --reroute_loop` process (tmux tpu-reroute) owns reroute
# + XM-truth reconcile, so the daemon must not double-run it. Env-scoped: only the
# tpu-daemon session sets =0; npu (no standalone reroute) keeps the default 1.
: "${TPU_ROUTE_INLANE_REROUTE:=1}"
# in-lane place pass (Step3): default 1 = daemon drains QUEUED->submit in the route
# lane (historical behavior). Set to 0 to SKIP it -- used when the standalone
# `route_check --dispatch_worker` (tpu-build-worker) is the SOLE drainer
# (router-dispatch + serial build in one loop), so the daemon must not also place
# (kills R1's two-drain-path incoherence). Env-scoped: only tpu-daemon sets =0;
# npu keeps default 1 (its build-worker is the classic --worker, daemon still places).
: "${TPU_ROUTE_INLANE_PLACE:=1}"
export TPU_LOCAL_QUEUE_FILE

cd /google/src/cloud/qiaos/run_amply_workspace/google3 || {
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
G3="/google/src/cloud/qiaos/run_amply_workspace/google3"
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
  # DURABLE OVERRIDE (money_check only): the xm_test blaze-bin read-path is on a
  # read-only objfs FUSE mount that holds an OLD (08-24) binary which HANGS, and
  # its srcfsd is wedged on negative lookups. Worse, this daemon cd's INTO the
  # wedged xm_test dir, and a par resolves runfiles relative to CWD -- so any
  # money_check launched from here D-state-hangs scanning the wedged CWD.
  #
  # money_check_wrapper.sh fixes both: it cd's to the HEALTHY run_amply workspace
  # and execs a money_check built there (live blaze-bin -> persistent local
  # execroot par -> self-heal rebuild), every probe timeout-guarded. Scope is
  # money_check ONLY; quota_check/infra_check keep their xm_test paths below.
  if [ "$name" = money_check ] && [ -x "$HOME/.tpu_bin/money_check_wrapper.sh" ]; then
    echo "$HOME/.tpu_bin/money_check_wrapper.sh"; return 0
  fi
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
  # ★PRIVATE SCRATCH + LOCKED PUBLISH. Two daemons on the same cache used to
  # share ONE `$CACHE.tmp` with no lock: the 165s `infra_check` of one could be
  # truncated by the other starting its own redirect, and whoever `mv`d last
  # won. A half-written cache is not merely a display bug -- budget_enforcer
  # prices the fleet from this file and would cancel against a partial fleet.
  # The pid suffix makes the scratch file unshareable; the lock covers only the
  # rename, so a slow checker never blocks the other daemon's publish.
  # ★$BASHPID, NOT $$: this function runs backgrounded (`run_infra_check &`),
  # and `$$` stays the PARENT shell's pid inside a subshell -- two concurrent
  # rounds would pick the SAME scratch name and overwrite each other, which is
  # the exact bug the suffix is meant to prevent. Verified: with `$$` the two
  # writers produced one file holding 5 lines of A followed by 15 of B.
  local tmp="${TMP_FILE}.${BASHPID:-$$}"
  "$bin" 2>/dev/null | grep -Ev "274311238|274310306|274303586|274276782|274276523|274275526|274274881|274274856|274269375|274256617|274256088|274255958|274456072|274454581" > "$tmp"
  if [ -s "$tmp" ]; then
    # Capture the subshell's status IMMEDIATELY: any statement in between --
    # including an echo -- resets `$?`.
    (
      flock -w 30 9 || exit 1
      mv "$tmp" "$CACHE_FILE"
      date +%s > "$TIME_FILE"
    ) 9>"${CACHE_FILE}.lock"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "$(date): Successfully updated infra cache"
    else
      # The previous cache is still intact and its age keeps ticking, which is
      # the honest signal; publishing a partial file would not be.
      echo "$(date): Error - infra cache publish failed (rc=$rc, lock busy >30s?);" \
           "previous cache kept. Retrying next round..."
      rm -f "$tmp"
    fi
  else
    rm -f "$tmp"
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
  if [ "$TPU_ROUTE_INLANE_PLACE" = "1" ]; then
    echo "$(date): route lane - place pass ($dry_flag, queue=$TPU_LOCAL_QUEUE_FILE, group_order=$TPU_ROUTE_GROUP_ORDER)"
    "$bin" --queue_file="$TPU_LOCAL_QUEUE_FILE" --group="$TPU_ROUTE_GROUP" --group_order="$TPU_ROUTE_GROUP_ORDER" "$dry_flag" 2>&1 \
      | sed 's/^/  [route:place] /'
  else
    echo "$(date): route lane - place pass SKIPPED (TPU_ROUTE_INLANE_PLACE=0; standalone tpu-dispatch-worker owns dispatch+build)"
  fi
  if [ "$TPU_ROUTE_INLANE_REROUTE" = "1" ]; then
    echo "$(date): route lane - reroute pass ($dry_flag, after ${TPU_ROUTE_REROUTE_AFTER_S}s)"
    "$bin" --queue_file="$TPU_LOCAL_QUEUE_FILE" --reroute \
      --reroute_after_s="$TPU_ROUTE_REROUTE_AFTER_S" "$dry_flag" 2>&1 \
      | sed 's/^/  [route:reroute] /'
  else
    echo "$(date): route lane - reroute pass SKIPPED (TPU_ROUTE_INLANE_REROUTE=0; standalone tpu-reroute owns reroute+reconcile)"
  fi
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
  #
  # TIMEOUT GUARD. A checker can hang inside its par launcher. money + quota
  # share a single `wait "$PID_QUOTA" "$PID_MONEY"` barrier in the round loop, so
  # ONE hung lane freezes the WHOLE round -- money.txt then ages past the 300s
  # staleness alarm and `tpu money`'s autoheal pointlessly restarts a daemon that
  # was never the problem. Bound every checker: SIGTERM at the deadline, SIGKILL
  # 10s later. A hung lane fails its own round and the barrier still clears, so
  # the other lane keeps refreshing every round.
  #
  # ★DEADLINE IS 300s, NOT 120s, AND THE MESSAGE NAMES NO CAUSE. Both changes
  # come from the same incident (2026-08-31): money_check takes ~51s of wall
  # clock on an IDLE machine, so 120s left barely 2.4x of headroom and every
  # busy round tripped it. The message used to read "likely the xm_test srcfs
  # wedge" -- a guess hardcoded into the log line, naming a workspace that no
  # longer exists on this machine. Three shifts read it as a measurement and
  # hunted a filesystem wedge; the real cause was four callers each running
  # their own copy of an identical ~51s job (fixed in money_check_wrapper.sh,
  # which now shares one run). A log line must report WHAT HAPPENED and where
  # to look; the moment it asserts WHY, it is a hypothesis wearing the clothes
  # of evidence, and it will outlive the condition that inspired it.
  out=$(timeout -k 10 "${CHECKER_TIMEOUT_S:-300}" "$bin" 2>&1)
  rc=$?
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    echo "$(date): Error - $2 check TIMED OUT (killed after ${CHECKER_TIMEOUT_S:-300}s; cause NOT diagnosed by this line -- check the checker's own stderr above, and \`pgrep -a money_check\` for a concurrent copy). Round continues."
    return $rc
  fi
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

  # RE-ESTABLISH THE CWD EVERY ROUND.
  #
  # The `cd` at line 42 runs once at startup. When the xm_test srcfs mount is
  # remounted underneath us -- which happens -- the handle this process holds
  # goes stale, `/proc/PID/cwd` starts reading `/cloud/...(deleted)`, and EVERY
  # CHILD INHERITS IT. Python dies during interpreter start, before any of our
  # code runs:
  #
  #     File "/<embedded stdlib>/sysconfig/__init__.py", line 198
  #       _PROJECT_BASE = _safe_realpath(os.getcwd())
  #     OSError: [Errno 107] Transport endpoint is not connected
  #
  # Measured 2026-08-25/26, three separate outages from this one cause: the
  # infra lane froze the check cache for 12 hours; a restart fixed it; then the
  # router lane inherited the same dead CWD and parked SEVEN queue entries as
  # HELD ("build produced no XID") -- entries that had nothing wrong with them
  # and that nothing retries once parked.
  #
  # A `cd` to the absolute path costs one syscall and re-opens a live handle,
  # so a remount now costs at most one round instead of poisoning the daemon
  # until someone notices and restarts it. Failure is non-fatal on purpose: if
  # the mount is down right now, the round should still try (money/quota read
  # their own paths and may well succeed) rather than the daemon exiting.
  cd /google/src/cloud/qiaos/run_amply_workspace/google3 2>/dev/null || {
    # srcfs mount is wedged right now. Do NOT stay on the poisoned CWD
    # (/cloud/...(deleted)) -- every Python child would die at interpreter
    # start on os.getcwd() (Errno 107). Fall back to local ext4 ($HOME) so the
    # round's children at least start; money/quota read their own absolute
    # paths and can still succeed. This mirrors the patch-3 getcwd guard in
    # tpu_wrapper.sh. Added 2026-08-26 after the route lane parked the board
    # stale ~24h on this exact crash.
    cd "$HOME" 2>/dev/null || cd /
    echo "$(date): WARN - cannot cd to xm_test/google3 (mount wedged?); fell back to CWD=$(pwd) so children can still start"
  }

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

  # Poll cadence between fast-lane rounds. money/quota only need ~2-minute
  # freshness (the user-facing staleness alarm in tpu_wrapper.sh is 300s), and
  # each round is seconds of wall time, so a 20s spin was needlessly hot: it
  # spawned money_check ~3x/min, adding objfs churn + build pressure on a shared
  # host. 120s keeps worst-case cache age (~interval + round wall ~= 135-180s)
  # comfortably under the 300s alarm while cutting checker spawns ~6x. Env-
  # overridable. (Autoheal judges liveness by ROUND PROGRESS, not cache age, so
  # a slower cadence does not trip it.)
  sleep "${TPU_DAEMON_POLL_SEC:-120}"
done
