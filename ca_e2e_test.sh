#!/bin/bash
# End-to-end test of the content-addressed publish CLI + real blaze cache.
#
# Drives the REAL `ca_stage.py publish` command (what tpu_wrapper.sh now calls
# when TPU_CONTENT_ADDRESSED_STAGE=1), against a THROWAWAY parent kept OUT of the
# live eqr_jax_final_stages tree, then actually builds the published target and
# proves the cache behaviour end to end:
#
#   P1 publish source            -> creates eqr_run_ca_<sha>, reused=0 fallback=0
#   B1 build //...ca_<sha>:main   -> COLD (link runs)
#   P2 publish IDENTICAL source   -> reused=1 (same dir, no new dir, no mutation)
#   B2 build again                -> CACHE HIT (fast, no link)
#
# Safety: throwaway parent experimental/qiaos/ca_e2e_stages (NOT the live staging
# parent), read-only source, host build lock held around builds, trap cleanup.
set -u
WS=/google/src/cloud/qiaos/run_amply_workspace/google3
SRC="$WS/experimental/qiaos/eqr_jax_final_stages/eqr_run_260915_220821_7777c3"  # real, read-only
PARENT_REL=experimental/qiaos/ca_e2e_stages
PARENT="$WS/$PARENT_REL"
CA=~/work/tpu_cmd/ca_stage.py
BLAZE="$HOME/.tpu_bin/shims/blaze"
BUILD_FLAGS=(--define=PYTYPE=FALSE --norun_validations)
LOG="${1:-/tmp/ca_e2e_test.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== CA E2E $(date -u +%FT%TZ) ==="
echo "  source(ro): $SRC"
echo "  parent(tw): $PARENT"

cleanup(){ echo "--- cleanup"; rm -rf "$PARENT" 2>/dev/null; [ -d "$PARENT" ] && echo "  STILL THERE" || echo "  gone"; }
trap cleanup EXIT

[ -d "$SRC" ] && [ -f "$SRC/BUILD" ] || { echo "FATAL: source missing/incomplete"; exit 1; }
rm -rf "$PARENT"; mkdir -p "$PARENT"

# ---- publish under the stage lock (rsync draws the CreateSnapshot bucket) ----
exec 200>"/tmp/tpu_stage.$(id -u).lock"; flock -w 300 200 || { echo "stage lock busy"; exit 1; }
echo; echo ">>> P1 publish (first time)"
OUT1=$(python3 "$CA" publish --source "$SRC" --parent "$PARENT" --rel-prefix "$PARENT_REL" --xm-launcher "$HOME/work/tpu_cmd/xm_launcher.py"); RC1=$?
echo "  rc=$RC1"; echo "  $OUT1"
flock -u 200; exec 200>&-
[ "$RC1" -eq 0 ] || { echo "FATAL: P1 failed"; exit 1; }
DIR1=$(echo "$OUT1" | grep -oE 'dir=[^ ]+' | cut -d= -f2-)
REL1=$(echo "$OUT1" | grep -oE 'rel=[^ ]+' | cut -d= -f2-)
REUSED1=$(echo "$OUT1" | grep -oE 'reused=[0-9]' | cut -d= -f2-)
echo "  DIR1=$DIR1"
echo "  basename check (must be eqr_run_ca_*): $(basename "$DIR1")"
echo "  marker: $(cat "$DIR1/.stage_ready" 2>/dev/null | head -c 20)..."
echo "  TARGET_LABEL in config.sh: $(grep TARGET_LABEL "$DIR1/config.sh")"
[ "$REUSED1" = "0" ] && echo "  [ok] first publish reused=0" || echo "  [FAIL] expected reused=0"

# ---- builds under the host build lock ----
exec 201>"/tmp/tpu_build.host.lock"; flock -w 1800 201 || { echo "build lock busy"; exit 1; }

build(){ local lbl="$1" tgt="$2" t0 out; echo; echo ">>> $lbl  $tgt"; t0=$(date +%s)
  out=$(cd "$WS" && "$BLAZE" build "${BUILD_FLAGS[@]}" "$tgt" 2>&1)
  local rc=$?; echo "$out" | grep -iE 'Elapsed time|Critical Path|Linking .*main|actions? (cached|ran)|ERROR:|FAILED' | sed 's/^/    /'
  local link=no; echo "$out" | grep -qiE 'Linking .*(unstamped_)?main' && link=YES
  echo "    ---> wall=$(($(date +%s)-t0))s rc=$rc link=$link"; return $rc
}

build "B1 COLD build" "//$REL1:main" || { echo "FATAL: B1 build failed"; flock -u 201; exit 1; }

echo; echo ">>> P2 publish (identical content again) -- expect reused=1"
exec 200>"/tmp/tpu_stage.$(id -u).lock"; flock -w 300 200
BEFORE=$(ls "$PARENT")
OUT2=$(python3 "$CA" publish --source "$SRC" --parent "$PARENT" --rel-prefix "$PARENT_REL" --xm-launcher "$HOME/work/tpu_cmd/xm_launcher.py")
flock -u 200; exec 200>&-
echo "  $OUT2"
DIR2=$(echo "$OUT2" | grep -oE 'dir=[^ ]+' | cut -d= -f2-)
REUSED2=$(echo "$OUT2" | grep -oE 'reused=[0-9]' | cut -d= -f2-)
AFTER=$(ls "$PARENT")
[ "$REUSED2" = "1" ] && [ "$DIR2" = "$DIR1" ] && echo "  [ok] identical content reused the SAME dir (reused=1)" || echo "  [FAIL] expected reuse of $DIR1, got dir=$DIR2 reused=$REUSED2"
[ "$BEFORE" = "$AFTER" ] && echo "  [ok] no new dir created on reuse" || { echo "  [FAIL] parent listing changed:"; diff <(echo "$BEFORE") <(echo "$AFTER"); }

build "B2 REBUILD (expect CACHE HIT, no link)" "//$REL1:main"

flock -u 201; exec 201>&-
echo; echo "=== CA E2E DONE $(date -u +%FT%TZ) ==="
echo "PASS criteria: P1 reused=0; B1 link=YES; P2 reused=1 & same dir & no new dir; B2 link=no & fast"
