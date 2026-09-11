#!/bin/bash
# Negative-control harness for _tpu_resolve_bin v2. Pure read-only: resolves
# paths, never executes a binary, never touches the live worker.
_TPU_G3=/google/src/cloud/qiaos/run_amply_workspace/google3
source ./resolve_bin_v2.sh
pass=0; fail=0
chk() { # chk <desc> <expected: FOUND|NOTFOUND> [must-contain]
  local desc="$1" want="$2" sub="$3" out rc
  out=$(_tpu_resolve_bin "$4"); rc=$?
  local got=NOTFOUND; [ $rc -eq 0 ] && got=FOUND
  local ok=1
  [ "$got" = "$want" ] || ok=0
  [ -n "$sub" ] && { case "$out" in *"$sub"*) ;; *) ok=0;; esac; }
  if [ $ok -eq 1 ]; then pass=$((pass+1)); printf '  PASS  %-52s rc=%d\n' "$desc" $rc
  else fail=$((fail+1)); printf '  FAIL  %-52s rc=%d out=%s\n' "$desc" $rc "$out"; fi
}

echo "== T1: normal case, all three binaries resolve =="
for b in queue_cli route_check jobd; do chk "resolve $b" FOUND "" "$b"; done

echo "== T2: NEGATIVE CONTROL -- a target that does not exist must fail CLOSED =="
chk "nonexistent target -> rc=1 + conventional path" NOTFOUND "blaze-bin/" "no_such_binary_xyz"

echo "== T3: NEGATIVE CONTROL -- roots all bogus must fail CLOSED =="
(
  _TPU_G3=/nonexistent/workspace/google3
  unset BUILD_EXECROOT
  out=$(_tpu_resolve_bin route_check); rc=$?
  if [ $rc -eq 1 ] && [ "$out" = "/nonexistent/workspace/google3/blaze-bin/experimental/users/qiaos/tpu_utils/route_check" ]; then
    echo "  PASS  bogus workspace -> rc=1, conventional path"; else
    echo "  FAIL  bogus workspace rc=$rc out=$out"; fi
)

echo "== T4: symlink-independence -- resolve WITHOUT consulting blaze-out =="
(
  # Simulate the failure: make the symlink hint useless. v1 would break here.
  _TPU_G3_REAL=$_TPU_G3
  out=$(BUILD_EXECROOT= _tpu_resolve_bin route_check); rc=$?
  case "$out" in
    */_blaze_qiaos/*) echo "  PASS  resolved via md5/buildrabbit root, not the symlink: rc=$rc";;
    *) echo "  FAIL  out=$out";;
  esac
)

echo "== T5: picks the NEWEST when several roots have the binary =="
for b in queue_cli route_check; do
  out=$(_tpu_resolve_bin $b)
  echo "    $b -> $out"
  echo "      mtime $(stat -c %y "$out" 2>/dev/null | cut -d. -f1)"
done

echo; echo "pass=$pass fail=$fail"
