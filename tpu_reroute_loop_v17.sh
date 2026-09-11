#!/bin/bash
# Standalone reconcile+reroute loop (tpu-reroute), infra-v17 2026-08-30.
#
# WHY THIS EXISTS: the daemon prints "reroute pass SKIPPED (standalone
# tpu-reroute owns reroute+reconcile)" every round -- but no such process was
# ever started. Result: 63 zombie SUBMITTED rows, oldest pending 179h.
#
# ONE-INSTANCE: flock, so a second start is a no-op rather than a second writer.
# MEMORY WALL: MemoryMax alone does NOT hold -- it excludes swap, so the process
#   swaps out instead of dying and the resulting PSI is what triggers
#   systemd-oomd to kill the whole tmux scope (31 processes, 20:07Z).
#   MemorySwapMax=0 is what makes the wall real (measured: 512MB probe survives
#   MemoryMax=64M with rc=0, dies rc=137 once MemorySwapMax=0 is added).
#
# FULL LOOP (reconcile + reroute) -- operator ordered it ON 21:34Z: "开,肯定要
# 开。这是规矩。" Earlier I ran reconcile-only out of a fear that turned out to
# be WRONG: I told the operator "开 = elt 三辆备胎没了". mark_reroute does NOT
# destroy a car -- it resets it to QUEUED (xid/cell cleared, cell cooled) so the
# router re-places it. Cancel-then-requeue, not delete.
#
# Prerequisite shipped with this switch: mark_reroute now preserves the old XID
# in prior_xids (route_lib.py). Without that, every re-route silently orphaned
# its cancelled experiment -- xid_recon reads prior_xids to tell a known car
# from a ghost, so re-routing WAS a ghost-car factory.
# 2026-09-06: verified clip_probe build includes queue snapshot conflict protection.
BIN='/usr/local/google/_blaze_qiaos/bb5e05891304127daf0b480f4298d971_buildrabbit/execroot/google3/blaze-out/k8-fastbuild/bin/experimental/users/qiaos/tpu_utils/route_check'
QUEUE="$HOME/.tpu_local_queue.json"
LOG="$HOME/work/.monitor_watch/tpu_reroute_loop_v17.log"

exec 9>/tmp/tpu-reroute-loop.lock
flock -n 9 || { echo "[reroute-loop] another instance holds the lock; exiting."; exit 0; }

# ★Every line gets a UTC timestamp. The router's own stdout carried none, so a
# log that exists to answer "how many cars did we re-route in the last hour"
# could not answer it: computing an hourly rate off this file yielded numbers
# parsed out of unrelated text (measured 2026-09-01 -- two plausible-looking
# values, 58 and 32, that were pure artefact). A rate needs a clock, and the
# cheapest place to put one is here, in the pipe, rather than in the binary.
# ★Do NOT wrap the binary in `stdbuf`: it is a GRTE binary, and stdbuf's
# LD_PRELOAD shim makes it load the SYSTEM libc, which dies with
# "GLIBC_2.38 not found" (measured 2026-09-01 -- the loop then exits rc=0
# every 30s and the log fills with restarts that look like normal rounds).
# Timestamping in the read loop is enough; bash `read` is unbuffered.
while true; do
  # 2026-09-06 16:25Z-18:2xZ: the nominal-RUNNING guard ran with its grace
  # pushed to 999999999 (i.e. off) for two hours. Kept here as the reason the
  # flag is now ABSENT rather than set to the default -- do not "restore" it.
  # That guard cancels an XM-RUNNING row when Borg reports no VM group in RUN
  # *and* nothing was ever written. Its disk half used to be blind to RESUMED
  # jobs: CnsOutputProbe.latest_mtime globbed `xid_<new xid>_*`, but a resume
  # keeps writing the directory it loaded from and never opens one of its own,
  # so the probe always answered "nothing was ever written" and a single Borg
  # sample was left deciding life and death. It cancelled elt_8n4l_v13a at
  # 11:22:51, four minutes after that job wrote checkpoint step 345000 and 85
  # seconds after its last tfevents write.
  # Fixed by the load_from fallback in latest_mtime (clip_probe build, 125
  # tests, mutation-verified: disabling the fallback fails 2). The guard is
  # back on its 3600s default and now has two independent votes again.
  systemd-run --user --scope -q \
      -p MemoryMax=8G -p MemorySwapMax=0 \
      "$BIN" --queue_file="$QUEUE" --reroute_loop --nodry_run --auto_resume_pruned 2>&1 \
      | while IFS= read -r _line; do
          printf '%s %s\n' "$(date -u +%FT%TZ)" "$_line"
        done >> "$LOG"
  echo "$(date -u +%FT%TZ) [reroute-loop] loop EXITED rc=$?; restarting in 30s" >> "$LOG"
  sleep 30
done
