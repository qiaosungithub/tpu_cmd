#!/bin/bash
# Controlled cache experiment for content-addressed stagedirs (design §9 step 1).
#
# Proves the cache hypothesis WITHOUT changing anything in the live launcher:
#   1. COLD  build a stable target over a copied source           -> baseline
#   2. REBUILD the SAME target, identical content                 -> expect CACHED (<40s)
#   3. build a DIFFERENT target name, IDENTICAL content           -> expect RELINK (the
#      current production scenario: unique dir name every launch defeats the cache)
#   4. change ONE configs/*.yml byte, rebuild target A            -> expect RELINK (config
#      is a .par input, so distinct content must get a distinct build)
#
# Safety:
#   * Throwaway package under experimental/qiaos/ca_stage_exp/ -- OUTSIDE the
#     eqr_jax_final_stages staging parent, so NO staging guard / reaper / GC ever
#     touches it, and it can NEVER collide with a live eqr_run_* stagedir.
#   * Copies are done under the per-user STAGE lock (fd 200) -- same CreateSnapshot
#     token bucket the live worker uses.
#   * Builds run under the host BUILD lock (fd 201) AND through the blaze shim's
#     own host_heavy lock, so they are serialized with any live worker build
#     (shared output_base -> serialize or risk found[] zombies).
#   * Reads a COMPLETE existing stagedir as the source; never writes to it.
#   * trap cleans up the throwaway package on exit.
set -u

# run_amply_workspace: HEALTHY (.citc/dropped_resources.ascii == 49 bytes header-
# only). clip_probe was dropping writes (5.7 MB of dropped-resource blocks) on
# 2026-09-15, which broke the first attempt mid-rsync; do not stage into it.
WS=/google/src/cloud/qiaos/run_amply_workspace/google3
# A COMPLETE, STABLE source: 1160 files, mtime 22:08, older than the worker's
# active dir so it is not being written while we read it.
SRC="$WS/experimental/qiaos/eqr_jax_final_stages/eqr_run_260915_220821_7777c3"
EXP_REL=experimental/qiaos/ca_stage_exp
EXP_ABS="$WS/$EXP_REL"
LOG="${1:-/tmp/ca_cache_experiment.log}"
BLAZE="$HOME/.tpu_bin/shims/blaze"
BUILD_FLAGS=(--define=PYTYPE=FALSE --norun_validations)

exec > >(tee -a "$LOG") 2>&1
echo "==================================================================="
echo "CA CACHE EXPERIMENT  $(date -u +%FT%TZ)"
echo "  source : $SRC ($(find "$SRC" -type f 2>/dev/null | wc -l) files)"
echo "  exp pkg: $EXP_ABS"
echo "  log    : $LOG"
echo "==================================================================="

cleanup() {
  echo "--- cleanup: removing throwaway package $EXP_ABS"
  rm -rf "$EXP_ABS" 2>/dev/null
  echo "--- cleanup done; $( [ -d "$EXP_ABS" ] && echo 'STILL PRESENT (check manually)' || echo 'gone' )"
}
trap cleanup EXIT

[ -d "$SRC" ] && [ -f "$SRC/BUILD" ] && [ -f "$SRC/main_eqr.py" ] || {
  echo "FATAL: source stagedir incomplete: $SRC"; exit 1; }

# ---- copy phase, under the per-user STAGE lock -------------------------------
echo; echo ">>> COPY PHASE (under per-user stage lock /tmp/tpu_stage.$(id -u).lock)"
exec 200>"/tmp/tpu_stage.$(id -u).lock"
if flock -w 300 200; then echo "    stage lock acquired"; else echo "    stage lock busy >300s, aborting"; exit 1; fi
rm -rf "$EXP_ABS"; mkdir -p "$EXP_ABS/exp_A" "$EXP_ABS/exp_B"
# rsync with the SAME excludes staging uses, so the copy mirrors a real stage.
RSEXC=(--exclude={'bazel-*','.citc','.git','.jj','.venv','__pycache__','*.npy','*.npz','*.ckpt','*.pth','*.pt','*.safetensors','data','logs','wandb'})
rsync -aL "${RSEXC[@]}" "$SRC/" "$EXP_ABS/exp_A/"
rsync -aL "${RSEXC[@]}" "$SRC/" "$EXP_ABS/exp_B/"
sync
flock -u 200; exec 200>&-
echo "    exp_A files=$(find "$EXP_ABS/exp_A" -type f | wc -l)  exp_B files=$(find "$EXP_ABS/exp_B" -type f | wc -l)"

# Content-hash both copies: they MUST be equal (identical content -> would share
# one content-addressed dir). This is the invariant the whole design rests on.
HA=$(python3 "$HOME/work/tpu_cmd/stage_content_hash.py" "$EXP_ABS/exp_A")
HB=$(python3 "$HOME/work/tpu_cmd/stage_content_hash.py" "$EXP_ABS/exp_B")
echo "    content hash exp_A=${HA:0:12}  exp_B=${HB:0:12}  $( [ "$HA" = "$HB" ] && echo 'EQUAL (good)' || echo 'DIFFER (unexpected!)')"

run_build() {  # $1=label  $2=target
  local label="$1" tgt="$2" t0 t1 out
  echo; echo ">>> BUILD [$label]  $tgt   $(date -u +%T)"
  t0=$(date +%s)
  out=$(cd "$WS" && "$BLAZE" build "${BUILD_FLAGS[@]}" "$tgt" 2>&1)
  t1=$(date +%s)
  # Echo just the signal lines; full output is in the log via tee already? No --
  # $out was captured, so print it, then the parsed summary.
  echo "$out" | grep -iE 'Elapsed time|Critical Path|actions? (cached|ran)|ObjFS|Linking|processes:|ERROR|FAILED' | sed 's/^/    /'
  local elapsed crit link
  elapsed=$(echo "$out" | grep -oE 'Elapsed time: [0-9.]+s' | head -1)
  crit=$(echo "$out"    | grep -oE 'Critical Path: [0-9.]+s' | head -1)
  if echo "$out" | grep -qiE 'Linking .*(unstamped_)?main'; then link="LINK RAN"; else link="no link action shown"; fi
  echo "    ---> wall=$((t1-t0))s  ${elapsed:-?}  ${crit:-?}  [$link]"
}

# ---- build phase, under the host BUILD lock ---------------------------------
echo; echo ">>> BUILD PHASE (under host build lock /tmp/tpu_build.host.lock)"
exec 201>"/tmp/tpu_build.host.lock"
if flock -w 1800 201; then echo "    host build lock acquired (serial with live worker)"; else echo "    host build lock busy >1800s, aborting"; exit 1; fi

run_build "1 COLD  exp_A"                 "//$EXP_REL/exp_A:main"
run_build "2 REBUILD exp_A (identical)"   "//$EXP_REL/exp_A:main"
run_build "3 exp_B (diff name, same content)" "//$EXP_REL/exp_B:main"

echo; echo "    mutating one configs/*.yml byte in exp_A ..."
YML=$(find "$EXP_ABS/exp_A/configs" -name '*.yml' 2>/dev/null | head -1)
if [ -n "$YML" ]; then echo "# ca-exp cache-bust $(date +%s)" >> "$YML"; echo "    appended a comment to ${YML#$EXP_ABS/}";
else echo "    (no yml found under exp_A/configs; skipping step 4)"; fi
run_build "4 exp_A after yml change"      "//$EXP_REL/exp_A:main"

flock -u 201; exec 201>&-
echo; echo "==================================================================="
echo "EXPERIMENT DONE  $(date -u +%FT%TZ)"
echo "Interpretation:"
echo "  step 2 wall << step 1 wall, no link  => cache HITS on identical rebuild (fix works)"
echo "  step 3 wall ~= step 1 wall, LINK RAN => different NAME defeats cache (the bug)"
echo "  step 4 LINK RAN                       => config is a .par input (distinct content -> distinct build)"
echo "==================================================================="
