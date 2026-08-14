#!/usr/bin/env bash
# Regression test for the two-operator split in tpu_wrapper.sh.
#
#   bash scripts/test_operator_scope.sh
#
# One Unix account (`qiaos`) is driven by two people: sqa through `tpu`, lyy
# through `npu`. `npu` is `tpu` with four variables rebound, so every isolation
# property here is one `if` away from silently disappearing on the next edit --
# and each one failed in production before it was written down:
#
#   * the board unioned a per-operator registry with an ACCOUNT-WIDE cache, so
#     lyy's `npu check` listed all 33 of sqa's runs;
#   * `cancel` handed any XID straight to `xmanager stop`, so one mistyped
#     digit stopped the other operator's job;
#   * a stale quota cache made `npu quota` run `tmux kill-session -t
#     tpu-daemon`, killing sqa's daemon -- the frozen board that recovery path
#     exists to repair.
#
# Nothing here talks to XManager, tmux or the network: `xmanager` and `tmux`
# are shadowed by shell functions, which win over PATH lookup, and every file
# the wrapper touches is redirected into a temp dir. Safe to run any time.
set -u

cd "$(dirname "$0")/.."
WRAPPER="$PWD/tpu_wrapper.sh"
[ -f "$WRAPPER" ] || { echo "tpu_wrapper.sh not found next to scripts/"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fails=0
pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; echo "$2" | sed 's/^/          /'; fails=$((fails + 1)); }
want()     { case "$2" in *"$3"*) pass "$1";; *) fail "$1" "expected to contain: $3";; esac; }
want_not() { case "$2" in *"$3"*) fail "$1" "expected NOT to contain: $3";; *) pass "$1";; esac; }

# --- fixtures ------------------------------------------------------------
# lyy's registry: one normal job, plus one whose name is a multi-line
# traceback blob -- a real entry written by the pre-fix `--config=` regex,
# kept because the renderer must survive data it did not write.
python3 - "$TMP" <<'PY'
import json, os, sys
tmp = sys.argv[1]
json.dump({
    "111111": {"exp_name": "lyy-mine", "status": "SUBMITTED"},
    "444444": {"exp_name": "developer_build_xmanager\nSee go/xm-universal-defaults for more\nERROR:absl:RH search failed",
               "status": "SUBMITTED"},
}, open(os.path.join(tmp, "lyy_jobs.json"), "w"))
PY

# The cache is what infra_check writes: every experiment on the ACCOUNT.
printf '%s\n' \
  '│ 111111 │ running │ lyy-mine │ - │ 100 │ - │' \
  '│ 333333 │ running │ lyy-outside-registry │ - │ 200 │ - │' \
  '│ 222222 │ running │ sqa-private-run │ - │ 300 │ - │' \
  > "$TMP/cache.txt"

# Run a wrapper command with the registry/cache redirected, xmanager and tmux
# shadowed, and the operator scope set by $1 ("" = sqa, "lyy-" = lyy).
run() {
  local prefix="$1"; shift
  cp "$TMP/lyy_jobs.json" "$TMP/jobs.$$.json"
  TPU_JOBS_FILE="$TMP/jobs.$$.json" \
  TPU_CHECK_CACHE_FILE="$TMP/cache.txt" \
  TPU_JOB_NAME_PREFIX="$prefix" \
  TPU_CMD_NAME="${prefix:+npu}" \
  TPU_OPERATOR="${prefix:+lyy}" \
  XM_CALLS="$TMP/xmanager_calls" \
  bash -c '
    source "$0" >/dev/null 2>&1
    xmanager() { echo "$*" >> "$XM_CALLS"; return 0; }
    tmux()     { echo "tmux $*" >> "$XM_CALLS"; return 0; }
    : "${TPU_CMD_NAME:=tpu}"
    "$@"
  ' "$WRAPPER" "$@" 2>&1
}

echo "== board scope =="
out=$(run "lyy-" tpu check -a)
want     "scoped board keeps a job from the operator's own registry" "$out" "111111"
want     "scoped board keeps a job carrying the operator's name prefix" "$out" "333333"
want_not "scoped board drops the other operator's account-wide job" "$out" "222222"

out=$(run "" tpu check -a)
want "unscoped board still shows everything charged to the account" "$out" "222222"
want "unscoped board still shows the other operator's jobs too" "$out" "111111"

echo
echo "== name sanitisation =="
out=$(run "lyy-" tpu check -a)
want "a multi-line registry name still renders" "$out" "developer_build_xmanager"
# The symptom is not the name's TEXT -- the NAME column truncates at 30 chars,
# so asserting on a word from line 2 of the blob passes even when broken. It is
# the LAYOUT: an embedded newline ends the row early and dumps the rest of the
# name, then every remaining column, at column 0. Every legitimate line of this
# table is indented or starts with a box/ANSI header, so a line beginning with
# the fixture's second line is the leak itself.
leaked=$(printf '%s\n' "$out" | grep -c '^See ')
if [ "$leaked" -eq 0 ]; then
  pass "...on ONE line, not spilling the rest of the row to column 0"
else
  fail "...on ONE line, not spilling the rest of the row to column 0" \
       "$leaked line(s) of the name blob leaked out of the NAME column"
fi

echo
echo "== cancel ownership =="
rm -f "$TMP/xmanager_calls"
out=$(run "lyy-" tpu cancel 222222)
want     "scoped cancel refuses the other operator's XID" "$out" "not yours"
want_not "...and does not reach xmanager" "$(cat "$TMP/xmanager_calls" 2>/dev/null)" "222222"

rm -f "$TMP/xmanager_calls"
run "lyy-" tpu cancel 111111 >/dev/null
want "scoped cancel allows an XID from the operator's own registry" \
     "$(cat "$TMP/xmanager_calls" 2>/dev/null)" "111111"

rm -f "$TMP/xmanager_calls"
run "lyy-" tpu cancel 333333 >/dev/null
want "scoped cancel allows an XID carrying the operator's prefix" \
     "$(cat "$TMP/xmanager_calls" 2>/dev/null)" "333333"

rm -f "$TMP/xmanager_calls"
run "" tpu cancel 222222 >/dev/null
want "unscoped cancel keeps full control of the account" \
     "$(cat "$TMP/xmanager_calls" 2>/dev/null)" "222222"

rm -f "$TMP/xmanager_calls"
out=$(run "lyy-" tpu cancel 111111 222222)
want     "a mixed batch is refused whole" "$out" "not yours"
want_not "...so the owned XID is not stopped either" \
         "$(cat "$TMP/xmanager_calls" 2>/dev/null)" "111111"

echo
echo "== shared-daemon guard =="
rm -f "$TMP/xmanager_calls"
out=$(run "lyy-" _tpu_restart_check_daemon)
want     "scoped operator is told to ask the account owner" "$out" "will not restart it"
want_not "...and never touches the owner's tmux session" \
         "$(cat "$TMP/xmanager_calls" 2>/dev/null)" "kill-session"

rm -f "$TMP/xmanager_calls"
run "" _tpu_restart_check_daemon >/dev/null
want "unscoped auto-recovery still restarts the daemon" \
     "$(cat "$TMP/xmanager_calls" 2>/dev/null)" "kill-session -t tpu-daemon"

echo
if [ "$fails" -eq 0 ]; then
  echo "all checks passed"
else
  echo "$fails check(s) FAILED"
fi
exit "$fails"
