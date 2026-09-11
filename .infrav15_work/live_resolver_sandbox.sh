_tpu_bin_roots() {
  # Print candidate ".../blaze-out" roots, most-authoritative first, deduped.
  local seen="" r h
  _tpu_emit_root() {
    case ":$seen:" in *":$1:"*) ;; *) seen="$seen:$1"; echo "$1";; esac
  }
  [ -n "$BUILD_EXECROOT" ] && _tpu_emit_root "$BUILD_EXECROOT/blaze-out"
  h=$(printf '%s' "$_TPU_G3" | md5sum | cut -d' ' -f1)
  for r in "/usr/local/google/home/qiaos/work/tpu_cmd/.infrav15_work/fx2/_blaze/${h}_buildrabbit" \
           "/usr/local/google/home/qiaos/work/tpu_cmd/.infrav15_work/fx2/_blaze/${h}"; do
    _tpu_emit_root "$r/execroot/google3/blaze-out"
  done
  r=$(readlink "$_TPU_G3/blaze-out" 2>/dev/null) && [ -n "$r" ] && _tpu_emit_root "$r"
  unset -f _tpu_emit_root
}
_tpu_resolve_bin() {
  # $1 = path under .../tpu_utils, e.g. "route_check" or "preflight/router_cli"
  local rel="experimental/users/qiaos/tpu_utils/$1" root cfg cand best="" bestt=0 t
  for root in $(_tpu_bin_roots); do
    for cfg in k8-fastbuild k8-opt k8-fastbuild-cuda; do
      cand="$root/$cfg/bin/$rel"
      [ -x "$cand" ] || continue
      t=$(stat -c %Y "$cand" 2>/dev/null) || continue
      if [ "$t" -gt "$bestt" ]; then bestt="$t"; best="$cand"; fi
    done
  done
  [ -n "$best" ] && { echo "$best"; return 0; }
  # Nothing found anywhere: return the conventional path so the caller's own
  # "not built, run blaze build ..." message still names something sensible.
  echo "$_TPU_G3/blaze-bin/$rel"
  return 1
}
