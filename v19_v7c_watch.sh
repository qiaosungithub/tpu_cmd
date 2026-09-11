#!/bin/bash
# Watch elt v7c (XID 285188026) to completion, then harvest the FID.
# Finish signal per EVAL_PROTOCOL.md: samples50k_iter* count goes N -> 0
# (clear_saved_results wipes the split ON COMPLETION), AND/OR the car's
# tfevents grows past 78 bytes (78 = empty run, 623/1601 = has metrics).
# Numeric sort throughout: `| tail` on the raw listing is LEXICOGRAPHIC and
# reads iter9 as latest from iter10 onward (protocol §"tail lies past iter9").
set -u
XID=285188026
D=/cns/is-d/home/qiaos/eqr_data/logs/elt-dit/xid_282154744_20260822_062609_dit_prod_train_500k_v6p_prod
SD="$D/eval_files/ttl=2d/g3p5"
NOTIFY="$HOME/.amply/bin/amply_notify"    # AGENTS.md: release binary is ACL-blocked
SESSION="chatty-bot"
LOG="$HOME/work/.monitor_watch/v19_v7c_watch.log"
say() {  # send, and CAPTURE the rc -- a dropped rc makes a failure untraceable (§3bd)
  local m="$1" out rc
  out=$("$NOTIFY" "$SESSION" "$m" 2>&1); rc=$?
  echo "$(date -u +%FT%TZ) notify rc=$rc :: ${out:0:120}" >>"$LOG"
  [ $rc -eq 0 ] || echo "$(date -u +%FT%TZ) NOTIFY FAILED rc=$rc msg=${m:0:160}" >>"$LOG"
}
peak=0
while :; do
  ls=$(timeout 120 fileutil ls "$SD" 2>/dev/null)
  n=$(printf '%s\n' "$ls" | grep -c 'samples50k_iter')
  mx=$(printf '%s\n' "$ls" | grep -oE 'iter[0-9]+$' | sed 's/iter//' | sort -n | tail -1)
  mx=${mx:-0}
  echo "$(date -u +%FT%TZ) shards=$n max_iter=$mx peak=$peak" >>"$LOG"
  # tfevents for THIS car only -- three cars share this dir, so the group_<xid>
  # in the filename is the only thing that says which car wrote it.
  ev=$(timeout 120 fileutil ls -l "$D" 2>/dev/null | grep "group_${XID}\." | awk '{print $5}' | sort -n | tail -1)
  ev=${ev:-0}
  if [ "$peak" -ge 90 ] && [ "$n" -lt 10 ]; then
    say "elt v7c ($XID): shard split WIPED ($peak -> $n) = FINISH signal. Harvesting FID now."
    break
  fi
  if [ "$ev" -gt 200 ]; then
    say "elt v7c ($XID): tfevents grew to ${ev}B (>78B = has metrics). Harvesting FID now."
    break
  fi
  [ "$mx" -gt "$peak" ] && peak=$mx
  sleep 60
done
# Harvest: pull this car's tfevents down and parse with the CALIBRATED parser
# (field 8 TensorProto; simple_value returns 0 scalars without erroring).
mkdir -p /tmp/v7c_harvest && cd /tmp/v7c_harvest || exit 1
timeout 300 fileutil cp -f "$D"/*group_${XID}.* /tmp/v7c_harvest/ 2>>"$LOG"
res=$(timeout 300 python3 "$HOME/work/.elt_results/v6/fid_tensor.py" '*.v2' 2>&1 | tail -20)
echo "$res" >>"$LOG"
fid=$(printf '%s\n' "$res" | grep -oE 'samples50k_vs_imf_fid_256 +[0-9.]+' | awk '{print $2}')
ref=$(printf '%s\n' "$res" | grep -oE 'imf_ref_num_samples +[0-9.]+' | awk '{print $2}')
say "elt v7c ($XID) HARVEST: samples50k_vs_imf_fid_256=${fid:-NOT_FOUND} imf_ref_num_samples=${ref:-NOT_FOUND} (protocol requires 1281168 or the number is void). Paper 2.83, official-ckpt sanity 3.0322, v6a 2.982632."
