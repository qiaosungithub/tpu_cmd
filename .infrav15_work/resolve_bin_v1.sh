_tpu_resolve_bin() {
  local rel="experimental/users/qiaos/tpu_utils/$1" cand
  for cand in "$_TPU_G3/blaze-out/k8-fastbuild/bin/$rel" \
              "$_TPU_G3/blaze-out/k8-opt/bin/$rel" \
              "$_TPU_G3/blaze-out/k8-fastbuild-cuda/bin/$rel"; do
    [ -x "$cand" ] && { echo "$cand"; return 0; }
  done
  cand="$_TPU_G3/blaze-bin/$rel"
  [ -x "$cand" ] && { echo "$cand"; return 0; }
  echo "$_TPU_G3/blaze-bin/$rel"; return 1
}
