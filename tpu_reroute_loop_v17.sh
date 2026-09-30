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
BIN='/google/src/cloud/qiaos/run_amply_workspace/google3/blaze-bin/experimental/users/qiaos/tpu_utils/route_check'
if [[ ! -x "$BIN" ]]; then
  BIN='/usr/local/google/_blaze_qiaos/bb5e05891304127daf0b480f4298d971_buildrabbit/execroot/google3/blaze-out/k8-fastbuild/bin/experimental/users/qiaos/tpu_utils/route_check'
fi
QUEUE="$HOME/.tpu_local_queue.json"
LOG="$HOME/work/.monitor_watch/tpu_reroute_loop_v17.log"

# ★2026-09-18: RECOVER the systemd user-bus env on every (re)spawn. The
# interlock restarts this script via `setsid nohup bash`, and the interlock's
# OWN env carries no XDG_RUNTIME_DIR / DBUS_SESSION_BUS_ADDRESS (measured: the
# watchdog it respawned at 20:15:47Z had only $HOME in /proc/<pid>/environ).
# Without them `systemd-run --user --scope` below cannot reach the user bus: it
# fails every 30s with "Failed to connect to user scope bus", the worker never
# starts, and reconcile goes dark (measured 19:43->20:38Z, ~55min). The
# interlock cannot catch this -- it only checks that the flock is HELD, and this
# broken loop holds it while spinning. Re-derive both from the uid so ANY
# respawn (interlock, login shell, or manual) self-heals. Uses := default so a
# already-good value from a real login session is left untouched.
_uid="$(id -u)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${_uid}}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/${_uid}/bus}"

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
  # REROUTE PATIENCE (operator 2026-09-29: set both clocks explicitly, 900s).
  # Both flags are the TPU base; route_lib.GPU_PATIENCE_MULTIPLIER (2x) applies
  # on top for GPU rows, so 900 = 15 min for TPU, 30 min for GPU.
  #   --reroute_after_s                  XM still PENDING this long after submit
  #   --reroute_nominal_running_grace_s  XM RUNNING, but no Borg VM group RUN and
  #                                      nothing ever written
  # The binary defaults are 300s (so 10 min for GPU). That cancelled GPU jobs
  # partway through a normal start. Measured 2026-09-29 on fresh h100/b200 rows:
  # XM submit -> Borg job exists 4-7 min, Borg PENDING -> machines 0-7 min (sj
  # b200 is the slow cell), container -> first CNS write ~2 min, so the first
  # write lands 11-17 min after submit. Over the prior 24 h, 409 of 564
  # submissions were cancelled before they ever started.
  # An older note here said the default was 3600s. That was wrong: the default
  # is 300s, lowered from 1200 on 2026-09-21. Never disable the nominal guard
  # with a huge value (it ran at 999999999 once, on 2026-09-06). It is what
  # rescues a job XManager calls RUNNING while its VM groups never leave PENDING.
  # Its disk half follows a resumed job's load_from dir, so a resume that is
  # writing is not judged "nothing written".
  REROUTE_AFTER_S=900
  NOMINAL_RUNNING_GRACE_S=900
  systemd-run --user --scope -q \
      -p MemoryMax=8G -p MemorySwapMax=0 \
      "$BIN" --queue_file="$QUEUE" --reroute_loop --nodry_run --auto_resume_pruned \
      --reroute_after_s="$REROUTE_AFTER_S" \
      --reroute_nominal_running_grace_s="$NOMINAL_RUNNING_GRACE_S" 2>&1 \
      | while IFS= read -r _line; do
          printf '%s %s\n' "$(date -u +%FT%TZ)" "$_line"
        done >> "$LOG"
  echo "$(date -u +%FT%TZ) [reroute-loop] loop EXITED rc=$?; restarting in 30s" >> "$LOG"
  sleep 30
done
