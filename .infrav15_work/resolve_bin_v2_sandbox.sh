# --- _tpu_resolve_bin v2 (infra-v15) -----------------------------------------
# WHY: the v1 resolver's three "absolute output layers" were all written as
# "$_TPU_G3/blaze-out/<config>/bin/$rel", and blaze-out is ITSELF the symlink
# blaze rewrites on every build -- so the three layers are ONE layer wearing
# three hats, and its comment ("real directories: a concurrent build in another
# config cannot move them") is false. When a build runs with a different
# output_base against the same checkout, the symlink repoints and all three
# candidates miss together => every `tpu` subcommand answers "not built" while
# the binaries sit untouched in the other root, and a long-lived worker keeps
# running happily from the root it started under.
#
# FIX: never trust the symlink for existence. Enumerate the REAL output_base
# roots for this workspace and pick the newest candidate that actually exists.
# Roots come from three independent sources so no single rewrite can hide them:
#   1. $BUILD_EXECROOT           (set inside a buildrabbit-style environment)
#   2. md5(workspace_directory)  (blaze's own naming rule, verified 2026-08-30)
#      ...and its "_buildrabbit" sibling
#   3. the symlink               (last, as a hint for a layout we do not know)
# Fails CLOSED: if nothing exists anywhere, echo the conventional path and
# return 1, exactly like v1, so callers' "not built" message still names a path.

_tpu_bin_roots() {
  # Print candidate ".../blaze-out" roots, most-authoritative first, deduped.
  local seen="" r
  _emit() { case ":$seen:" in *":$1:"*) ;; *) seen="$seen:$1"; echo "$1";; esac; }
  [ -n "$BUILD_EXECROOT" ] && _emit "$BUILD_EXECROOT/blaze-out"
  local h; h=$(printf '%s' "$_TPU_G3" | md5sum | cut -d' ' -f1)
  for r in "/usr/local/google/home/qiaos/work/tpu_cmd/.infrav15_work/fx/_blaze/${h}_buildrabbit" \
           "/usr/local/google/home/qiaos/work/tpu_cmd/.infrav15_work/fx/_blaze/${h}"; do
    _emit "$r/execroot/google3/blaze-out"
  done
  r=$(readlink "$_TPU_G3/blaze-out" 2>/dev/null) && [ -n "$r" ] && _emit "$r"
  unset -f _emit
}

_tpu_resolve_bin() {
  local rel="experimental/users/qiaos/tpu_utils/$1" root cfg cand best="" bestt=0 t
  for root in $(_tpu_bin_roots); do
    for cfg in k8-fastbuild k8-opt k8-fastbuild-cuda; do
      cand="$root/$cfg/bin/$rel"
      [ -x "$cand" ] || continue
      t=$(stat -c %Y "$cand" 2>/dev/null) || continue
      if [ "$t" -gt "$bestt" ]; then bestt="$t"; best="$cand"; fi
    done
  done
  if [ -n "$best" ]; then echo "$best"; return 0; fi
  echo "$_TPU_G3/blaze-bin/$rel"
  return 1
}
