#!/bin/bash
# sync_router_to_workspace.sh
#
# Single-source-of-truth sync for the TPU smart-router source.
#
# WHY THIS EXISTS: route_check.py / route_lib.py are NOT committed to google3
# HEAD and were never tracked in this git repo either. They lived only as
# working-copy files, so every CitC client that ever touched them kept its own
# private copy. On 2026-09-16 that bit us: a concurrency fix (baseline-guarded
# merge_and_save_touched) was made in the `clip_probe` checkout, but the daemons
# build from `run_amply_workspace`, so the fix never reached production and a
# cleared-then-resurrected queue row survived for days.
#
# THE RULE NOW:
#   * EDIT the router in the git repo:  ~/work/tpu_cmd/google3_tpu_utils/
#   * RUN THIS SCRIPT to push it into the build/deploy checkout (run_amply_workspace)
#   * then `blaze test` + `blaze build` + restart the reroute-loop / dispatch-worker.
#
# SCOPE: this syncs ONLY the four router files below. It deliberately does NOT
# touch the other tpu_utils files (avail_provider.py, cell_locality.py,
# money_check.py, ...): as of 2026-09-16 the repo copies of several of those have
# drifted from the live checkout in BOTH directions, so a blanket copy would
# regress live code. Keep this script narrow until that drift is reconciled
# separately.
set -euo pipefail

REPO_DIR="${ROUTER_REPO_DIR:-$HOME/work/tpu_cmd/google3_tpu_utils}"
WS_DIR="${ROUTER_WS_DIR:-/google/src/cloud/qiaos/run_amply_workspace/google3/experimental/users/qiaos/tpu_utils}"

# The router files this repo is the source of truth for.
ROUTER_FILES=(route_check.py route_lib.py route_check_test.py route_lib_test.py)

echo "[sync-router] repo (source): $REPO_DIR"
echo "[sync-router] workspace (dest): $WS_DIR"

[ -d "$REPO_DIR" ] || { echo "[sync-router] FATAL: repo dir missing: $REPO_DIR" >&2; exit 1; }
[ -d "$WS_DIR" ]   || { echo "[sync-router] FATAL: workspace dir missing: $WS_DIR" >&2; exit 1; }

changed=0
for f in "${ROUTER_FILES[@]}"; do
  src="$REPO_DIR/$f"
  dst="$WS_DIR/$f"
  [ -e "$src" ] || { echo "[sync-router] FATAL: source file missing: $src" >&2; exit 1; }
  if [ -e "$dst" ] && cmp -s "$src" "$dst"; then
    echo "[sync-router]   = $f (already identical)"
  else
    cp -f "$src" "$dst"
    echo "[sync-router]   -> $f (updated $(wc -l < "$dst") lines)"
    changed=$((changed + 1))
  fi
done

# Sanity: every synced .py must import-compile, so a broken paste is caught here
# rather than at daemon start.
echo "[sync-router] py_compile check ..."
for f in route_check.py route_lib.py; do
  python3 -m py_compile "$WS_DIR/$f" \
    && echo "[sync-router]   ok: $f" \
    || { echo "[sync-router] FATAL: $f failed to compile after sync" >&2; exit 1; }
done

# BUILD is a shared file (many targets); do not clobber it. Just warn on drift so
# a route-target BUILD change in the repo is not silently left behind.
if [ -e "$REPO_DIR/BUILD" ] && [ -e "$WS_DIR/BUILD" ] && ! cmp -s "$REPO_DIR/BUILD" "$WS_DIR/BUILD"; then
  echo "[sync-router] NOTE: BUILD differs between repo and workspace (not auto-synced;"
  echo "[sync-router]       reconcile by hand if you changed a route_* target)."
fi

echo "[sync-router] done ($changed file(s) updated)."
if [ "$changed" -gt 0 ]; then
  cat <<EOF
[sync-router] NEXT:
  cd ${WS_DIR%/experimental/*}
  blaze test experimental/users/qiaos/tpu_utils:route_check_test \\
             experimental/users/qiaos/tpu_utils:route_lib_test --nocache_test_results
  blaze build experimental/users/qiaos/tpu_utils:route_check
  # then restart the reroute-loop / dispatch-worker so they exec the new binary:
  #   pkill -f 'route_check.*\.tpu_local_queue\.json.*reroute_loop'   # wrapper relaunches it
EOF
fi
