#!/bin/bash

# Centralized Group Mappings (Single Source of Truth)
# LINT.IfChange(group_map)
# Keep in sync with experimental/users/qiaos/tpu_utils/group_utils.py::GROUP_MAP.
get_alloc_by_group_id() {
  case "$1" in
    1) echo "group:deepmind-dynamic/gdm-resources-prod-shared-users-dynamic" ;;
    2) echo "group:deepmind-dynamic/gdm-viscam-goflow-dynamic" ;;
    3) echo "group:deepmind-dynamic/gdm-viscam-interns-dynamic" ;;
    4) echo "group:deepmind-dynamic/viscam-interns" ;;
    5) echo "group:deepmind-dynamic/vqfree-xm" ;;
    6) echo "group:dm/deepmind-large-scale-workshop" ;;
    7) echo "group:dm/dm-resources-prod-shared" ;;
    8) echo "group:gdm-aux/brain-vasp-shared-user-xm" ;;
    9) echo "group:deepmind-dynamic/fr-dna-grand-challenge-team-resource" ;;
    *) echo "" ;;
  esac
}
# LINT.ThenChange(//depot/google3/experimental/users/qiaos/tpu_utils/group_utils.py)

get_group_id_by_alloc() {
  local alloc="$1"
  case "$alloc" in
    *"gdm-resources-prod-shared-users-dynamic"*) echo "g1" ;;
    *"gdm-viscam-goflow-dynamic"*) echo "g2" ;;
    *"gdm-viscam-interns-dynamic"*) echo "g3" ;;
    *"viscam-interns"*) echo "g4" ;;
    *"vqfree-xm"*) echo "g5" ;;
    *"deepmind-large-scale-workshop"*) echo "g6" ;;
    *"dm-resources-prod-shared"*) echo "g7" ;;
    *"brain-vasp-shared-user-xm"*) echo "g8" ;;
    *"fr-dna-grand-challenge-team-resource"*) echo "g9" ;;
    *)
      if [[ "$alloc" == group:* ]]; then
        echo "${alloc#group:}"
      else
        echo "${alloc:-"-"}"
      fi
      ;;
  esac
}

# Preflight & router blaze targets (built once, reused). If missing, tpu queue
# will still work but skip the pre-flight check with a warning.
_PREFLIGHT_CLI_BIN="/google/src/cloud/qiaos/xm_test/google3/blaze-bin/experimental/users/qiaos/tpu_utils/preflight/preflight_cli"
_ROUTER_CLI_BIN="/google/src/cloud/qiaos/xm_test/google3/blaze-bin/experimental/users/qiaos/tpu_utils/preflight/router_cli"
_INFRA_CHECK_BIN="/google/src/cloud/qiaos/xm_test/google3/blaze-bin/experimental/users/qiaos/tpu_utils/infra_check"
_MACH_LOCALITY="/usr/local/bin/mach_locality"
# Default must track xm_launcher.py's --bucket default; only used to work out
# which continent the data is in when the caller does not pass --bucket.
_DEFAULT_BUCKET="/cns/yutulpz-d/home/qiaos/eqr_data"

_continent_of() {
  # Borg cell -> continent ('na', 'eu', 'ap'), or empty when unknown.
  [ -z "$1" ] && return 0
  [ -x "$_MACH_LOCALITY" ] || return 0
  "$_MACH_LOCALITY" -k continent "$1" 2>/dev/null | awk '{print $2}'
}

_check_locality() {
  # $1 = explicitly requested cell (may be empty), $2 = the passthrough args,
  # searched for --bucket. Warns; never blocks. Placement is a tradeoff the
  # caller may be making deliberately, and a guard that refuses to submit would
  # be worse than the stall it prevents.
  local cell="$1" args="$2"
  [ -z "$cell" ] && return 0

  local bucket="$_DEFAULT_BUCKET"
  case "$args" in
    *--bucket=*) bucket="${args##*--bucket=}"; bucket="${bucket%% *}" ;;
  esac
  # /cns/<cell>-d/... -> <cell>
  local data_cell="${bucket#/cns/}"; data_cell="${data_cell%%-d/*}"
  [ "$data_cell" = "$bucket" ] && return 0   # not a /cns/ path; nothing to compare

  local c_compute c_data
  c_compute=$(_continent_of "$cell")
  c_data=$(_continent_of "$data_cell")
  [ -z "$c_compute" ] && return 0
  [ -z "$c_data" ] && return 0

  if [ "$c_compute" != "$c_data" ]; then
    echo -e "\033[31m[locality] WARNING: compute and data are on different continents.\033[0m"
    echo -e "\033[31m  compute cell : $cell ($c_compute)\033[0m"
    echo -e "\033[31m  data bucket  : $data_cell ($c_data)\033[0m"
    echo -e "\033[33m  Every restart re-reads the checkpoint across an ocean. XID 275793223\033[0m"
    echo -e "\033[33m  was killed and restarted 26 times this way without training a step.\033[0m"
    echo -e "\033[33m  Pick a cell in '$c_data', or point --bucket at a '$c_compute' mirror.\033[0m"
  else
    echo -e "\033[2m[locality] ok: compute $cell and data $data_cell are both in $c_data\033[0m"
  fi
}

# ---------------------------------------------------------------------------
# Limit orders: cap what a launched job pays per chip-hour.
#
# WHY: without a cap, a price spike drains the team's shared credit balance at
# market rate. With one, the job pauses instead of overspending and auto-resumes
# when the price falls. Scope is per-XID, so this never touches teammates' jobs
# and is not overwritten by the periodic group-wide (MDB) push -- resolution
# order is SCU > XID > MDB, most granular wins.
#
# WHY set_limit_order AND NOT dynlo: dynlo computes a multiple of the historical
# median, which is the nicer semantic, but it calls
# QuotaMarketplaceDataService.GetLimitOrders, which a restricted LOAS credential
# cannot reach (go/loas-restricted-credentials). set_limit_order reaches the
# same state over a path that works today. Revisit if that access is granted.
#
# WHY a per-card table AND NOT one global number: --price is an ABSOLUTE price
# per chip-hour, and prices differ ~20x across cards. Measured 2026-07-29:
# v4 1.22, v5p 20.32, v6e 26.01, v6p 0-60 credits/chip-hour. A single number
# either fails to cap the cheap cards or permanently blocks the dear ones.
#
# The values below are roughly 3x the observed price -- loose enough to ride out
# ordinary intraday swings (this pool moves several fold within a day), tight
# enough to stop a genuine blowout. They are a blast-radius limit, not an
# attempt to track the market. Re-check with `tpu money` if prices shift.
_TPU_SLO_BIN="/google/bin/releases/brain-quota/set_limit_order/set_limit_order"

# Absolute cap in credits per CHIP-hour, by accelerator family.
_tpu_limit_price_for_arch() {
  case "$1" in
    v4)      echo "12"  ;;   # market ~1.2
    v5p)     echo "60"  ;;   # market ~20.3
    v5e)     echo "30"  ;;   # market low; conservative
    v6e)     echo "80"  ;;   # market ~26.0
    v6p)     echo "180" ;;   # market 0-60
    *)       echo "100" ;;   # unknown family: generous but still bounded
  esac
}

# _tpu_set_limit_order <xid> <tpu_type> <tier> [price_override]
_tpu_set_limit_order() {
  local xid="$1" tpu_type="$2" tier="$3" override="$4"

  # BATCH clears at 0 by construction (supply >= demand), and BATCH admission
  # never reads a floor or a price. A cap there would be inert.
  if [ "${tier^^}" = "BATCH" ]; then
    echo -e "\033[2m[limit order] Skipped: BATCH clears at price 0, a cap would do nothing.\033[0m"
    return 0
  fi

  if [ ! -x "$_TPU_SLO_BIN" ]; then
    echo -e "\033[33m[limit order] set_limit_order not found — skipping.\033[0m"
    return 0
  fi

  local arch="${tpu_type%%-*}"
  local price="${override:-$(_tpu_limit_price_for_arch "$arch")}"

  echo -e "\033[36m[limit order] Capping XID $xid at ${price} credits/chip-hour (${arch})...\033[0m"
  local lo_log="/tmp/tpu_limit_order_${xid}.log"
  if "$_TPU_SLO_BIN" --xid="$xid" --price="$price" > "$lo_log" 2>&1; then
    echo -e "\033[32m[limit order] Capped XID $xid at ${price} cr/chip-hr.\033[0m"
    echo -e "\033[2m  Change:  $_TPU_SLO_BIN --xid=$xid --price=<N>\033[0m"
    echo -e "\033[2m  Remove:  $_TPU_SLO_BIN --xid=$xid --price=-1\033[0m"
  else
    # Never fail a launch over this: the job is already submitted, and running
    # uncapped beats the user believing the launch broke.
    echo -e "\033[33m[limit order] Could not cap XID $xid (job still running, just uncapped).\033[0m"
    if grep -q "loas-restricted-credentials" "$lo_log" 2>/dev/null; then
      echo -e "\033[33m  Cause: restricted LOAS credential. Run 'gcert', then retry:\033[0m"
    fi
    echo -e "\033[2m  Retry:  $_TPU_SLO_BIN --xid=$xid --price=$price\033[0m"
    echo -e "\033[2m  Log:    $lo_log\033[0m"
  fi
  return 0
}

# Intercepts 'tpu queue' and forwards everything else to the check/quota logic or original tpu system command.
tpu() {
  if [ "$1" = "queue" ] || [ "$1" = "q" ]; then
    shift
    local orig_dir="$PWD"
    local group=""
    local tpu_type=""
    local priority=""
    local tier=""
    local power=""
    local force=0
    local skip_preflight=0
    local lo_price=""
    local no_limit_order=0
    local user_cell=""
    local resume_xid=""
    local passthrough_args=()
    
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --group=*)
          group="${1#*=}"
          shift
          ;;
        --group)
          group="$2"
          shift 2
          ;;
        --tpu_type=*)
          tpu_type="${1#*=}"
          shift
          ;;
        --tpu_type)
          tpu_type="$2"
          shift 2
          ;;
        --power=*)
          power="${1#*=}"
          shift
          ;;
        --power)
          power="$2"
          shift 2
          ;;
        --priority=*)
          priority="${1#*=}"
          shift
          ;;
        --priority)
          priority="$2"
          shift 2
          ;;
        --tier=*|--service_tier=*)
          tier="${1#*=}"
          shift
          ;;
        --tier|--service_tier)
          tier="$2"
          shift 2
          ;;
        --force|-f)
          force=1
          shift
          ;;
        --lo-price=*|--limit-order-price=*)
          lo_price="${1#*=}"
          shift
          continue
          ;;
        --no-limit-order|--no-lo)
          no_limit_order=1
          shift
          continue
          ;;
        --skip-preflight|--no-preflight)
          skip_preflight=1
          shift
          ;;
        # --app.<flag>=<v> forwards <flag> to the packaged binary verbatim.
        # The rest of this parser is an allowlist on purpose -- a mistyped flag
        # should be refused here, not silently ignored on Borg. But an allowlist
        # cannot know the flags of every binary this wrapper launches, and the
        # alternative (extend the list per job) means editing shared tooling for
        # a one-off. The `app.` prefix keeps the property that matters: a name
        # is only forwarded when it was ASKED to be forwarded, so a typo in a
        # wrapper flag still errors instead of vanishing.
        --app.*)
          passthrough_args+=("--${1#--app.}")
          shift
          ;;
        --resume_xid=*)
          # Remembered, not just forwarded: a resume must re-run the ORIGINAL
          # snapshot, so the staging step below has to know this is one.
          resume_xid="${1#*=}"
          passthrough_args+=("$1")
          shift
          ;;
        --exp_name=*|--config=*|--bucket=*|--workdir=*|--config.*)
          passthrough_args+=("$1")
          shift
          ;;
        # Checkpoint / resume selectors and the Borg restart budget. The
        # launcher turns load_from & wandb_resume_id into $LOAD_FROM /
        # $WANDB_RESUME_ID env vars (unified_infra convention), so they must
        # not be rewritten into --config.* flags here.
        --cell=*)
          # Remembered so the router never overrides an explicit choice.
          user_cell="${1#*=}"
          passthrough_args+=("$1")
          shift
          ;;
        --cell)
          user_cell="$2"
          passthrough_args+=("$1=$2")
          shift 2
          ;;
        # --tmp_ram_fs_gib sizes the per-task RAM disk backing /tmp. The
        # launcher has always accepted it; it just had no route through here,
        # so every job silently took the 16 GiB TPU-training default. On a
        # BATCH submit that default is charged as cell-wide shared RAM, and a
        # CPU-only job that needs kilobytes gets stuck behind it:
        #   "QUEUED: Not enough cell-wide shared resources for Best Effort
        #    paying batch collection. Resources exceeded by 16.12GiB RAM."
        # (XID 276768092, cell go). The ask, not the alloc, was the blocker.
        # --ram_gib sizes the per-task memory requirement. It is NOT the same
        # thing as --tmp_ram_fs_gib (the RAM disk behind /tmp). Omitted by
        # default, so Borg's own default still applies and no existing job
        # changes behaviour.
        # --replicas is the Borg TASK COUNT. Every job before it ran as exactly
        # one task, which is right for a TPU trainer (one work unit drives the
        # whole slice) and useless for an embarrassingly-parallel CPU job, where
        # the task count IS the parallelism. Absent => the launcher emits no
        # `replicas` requirement at all, so no existing job changes shape.
        # --autopilot / --noautopilot decides whether Borg Autopilot may resize
        # the job. It is ON by default in XManager, and for a batch job with a
        # MEASURED footprint that is actively harmful: it widens a 1-core ask to
        # min 1024 / max 40000 milligcu, and a best-effort job then finds no
        # machine and sits in DISABLED reporting what looks like a cell capacity
        # problem. Absent => XManager's own behaviour, unchanged.
        --load_from=*|--wandb_resume_id=*|--borg_max_task_failures=*|--borg_max_per_task_failures=*|--tmp_ram_fs_gib=*|--ram_gib=*|--replicas=*|--autopilot=*)
          passthrough_args+=("$1")
          shift
          ;;
        --autopilot|--noautopilot)
          passthrough_args+=("$1")
          shift
          ;;
        --load_from|--wandb_resume_id|--borg_max_task_failures|--borg_max_per_task_failures|--tmp_ram_fs_gib|--ram_gib|--replicas)
          passthrough_args+=("$1=$2")
          shift 2
          ;;
        --exp_name|-n|--config|--bucket|--workdir|--resume_xid)
          passthrough_args+=("$1" "$2")
          shift 2
          ;;
        -*)
          echo "Error: Unsupported option '$1'"
          return 1
          ;;
        *)
          echo "Error: Unexpected argument '$1'"
          return 1
          ;;
      esac
    done

    # Router mode: --power resolves to (group, tpu_type) automatically.
    if [ -n "$power" ]; then
      if [ -n "$tpu_type" ] || [ -n "$group" ]; then
        echo -e "\033[31mError: --power is mutually exclusive with --tpu_type / --group.\033[0m"
        return 1
      fi
      if [ ! -x "$_ROUTER_CLI_BIN" ]; then
        echo -e "\033[31mError: router binary not built. Run:\033[0m"
        echo "  (cd /google/src/cloud/qiaos/xm_test/google3 && blaze build experimental/users/qiaos/tpu_utils/preflight:router_cli)"
        return 1
      fi
      echo -e "\033[36m[tpu queue] Router mode: --power=$power. Probing candidates...\033[0m"
      local rtier="${tier:-PROD}"
      local reco_json
      # --explain so limit-order-blocked combos are still reported (flagged)
      # rather than vanishing: "everything is capped" and "nothing exists" need
      # different fixes. It costs nothing -- the probes run either way, --top
      # only truncates the result.
      reco_json=$("$_ROUTER_CLI_BIN" --power="$power" --tier="$rtier" --top=1 --json --explain 2>/dev/null)
      if [ -z "$reco_json" ] || [ "$reco_json" = "[]" ]; then
        echo -e "\033[31mRouter found no viable candidate. Falling back to the human view:\033[0m"
        "$_ROUTER_CLI_BIN" --power="$power" --tier="$rtier" --top=3 --explain
        return 1
      fi
      # Parse top-1 JSON. `cell`, `price` and `blocked` are newer keys; the
      # defaults keep this working against an older router binary that does
      # not emit them yet.
      local reco_data
      reco_data=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])[0]
price = d.get('price')
print(d['group'], d['tpu_type'], d['status'],
      d.get('cell') or '-',
      '-' if price is None else '%.2f' % price,
      '1' if d.get('blocked') else '0')" "$reco_json" 2>/dev/null)
      if [ -z "$reco_data" ]; then
        echo -e "\033[31mCould not parse router output.\033[0m"
        echo "$reco_json"
        return 1
      fi
      local reco_cell reco_price reco_blocked
      read -r group tpu_type reco_status reco_cell reco_price reco_blocked <<< "$reco_data"

      # A blocked top pick means every candidate was stopped by a limit order.
      # Submitting anyway just parks the job in the queue invisibly, so stop
      # and point at the explanation instead.
      if [ "$reco_blocked" = "1" ]; then
        echo -e "\033[31m[router] every candidate is blocked by a limit order (price cap).\033[0m"
        echo -e "\033[31m  A blocked job is dropped before any capacity check; free quota will not help.\033[0m"
        echo "  See:  tpu route --power=$power --tier=$rtier --explain"
        if [ "$force" != "1" ]; then
          echo -e "\033[31m  Refusing to submit. Re-run with --force to override.\033[0m"
          return 1
        fi
        echo -e "\033[33m  --force is set; submitting anyway.\033[0m"
      fi

      local reco_line="[router] recommended: group=$group tpu_type=$tpu_type (status=$reco_status)"
      # Pin the recommended cell -- but never override an explicit --cell.
      # GQM charges per cell, so this is what makes the cheap cell actually
      # take effect rather than just being printed.
      if [ -n "$user_cell" ]; then
        reco_line="$reco_line; keeping your --cell=$user_cell (router suggested $reco_cell)"
      elif [ -n "$reco_cell" ] && [ "$reco_cell" != "-" ]; then
        passthrough_args+=("--cell=$reco_cell")
        reco_line="$reco_line cell=$reco_cell @ ${reco_price} credits/chip-hr"
      else
        reco_line="$reco_line (no cell recommendation; letting the allocator choose)"
      fi
      echo -e "\033[32m${reco_line}\033[0m"
      # Auto-set tier if not set
      if [ -z "$tier" ]; then tier="$rtier"; fi
    fi
    
    if [ -z "$tpu_type" ]; then
      echo "Error: Please specify --tpu_type (e.g., v4-64,v5p-32) or --power="
      return 1
    fi

    if [ -z "$group" ]; then
      echo "Error: Please specify --group (e.g., --group=5) or --power="
      return 1
    fi

    # ============ PREFLIGHT CHECK (before the ~5 min bazel packaging) ============
    if [ "$skip_preflight" != "1" ] && [ -x "$_PREFLIGHT_CLI_BIN" ]; then
      local pf_tier="${tier:-PROD}"
      # Auto-apply the tier-inference rule (mirrors what happens later).
      local first_g="${group%%,*}"
      if [ "$first_g" = "5" ] && [ -z "$tier" ]; then pf_tier="PROD"; fi
      echo -e "\033[36m[tpu queue] Running preflight check on ${first_g}/${tpu_type} @ ${pf_tier}...\033[0m"
      "$_PREFLIGHT_CLI_BIN" --tpu_type="$tpu_type" --group="$first_g" --tier="$pf_tier"
      local pf_status=$?
      if [ "$pf_status" -eq 1 ]; then
        if [ "$force" = "1" ]; then
          echo -e "\033[33m[tpu queue] Preflight said RED but --force is set; continuing anyway.\033[0m"
        else
          echo -e "\033[31m[tpu queue] REFUSING to submit: preflight verdict = RED.\033[0m"
          echo -e "\033[31m  If you believe this is a false alarm, re-run with --force to override.\033[0m"
          echo -e "\033[31m  Or use --skip-preflight to bypass the check entirely.\033[0m"
          return 1
        fi
      elif [ "$pf_status" -eq 2 ]; then
        echo -e "\033[33m[tpu queue] Preflight crashed (exit=2); proceeding without pre-check.\033[0m"
      else
        # exit 0 covers both GREEN and YELLOW; the CLI already printed the warnings
        :
      fi
    elif [ "$skip_preflight" != "1" ]; then
      echo -e "\033[33m[tpu queue] Preflight binary not built at $_PREFLIGHT_CLI_BIN — skipping.\033[0m"
      echo -e "\033[33m  Build with: (cd /google/src/cloud/qiaos/xm_test/google3 && blaze build experimental/users/qiaos/tpu_utils/preflight:preflight_cli)\033[0m"
    fi
    
    # Default tier if not specified (we handle this externally or keep it empty for defaults)
    if [ -z "$tier" ]; then
      tier=""
    fi
    
    local now=$(date '+%y%m%d_%H%M%S')
    local logdir="$HOME/logs/eqr_run_${now}"
    mkdir -p "$logdir"
    echo "Created logdir: $logdir"
    
    local stagedir="experimental/qiaos/eqr_jax_final_stages/eqr_run_${now}"
    local abs_stagedir="/google/src/cloud/qiaos/EqR-jax/google3/${stagedir}"

    # A RESUME RE-RUNS THE ORIGINAL SNAPSHOT. NEVER THE CURRENT CHECKOUT.
    #
    # Packaging the working tree instead is how a resume ends up running code
    # the checkpoint has never seen. Two ways it bites, both observed:
    #
    #   * the config dialect moves on. `training.epochs` / `max_steps` /
    #     `train_epochs_per_iter` were retired in favour of `total_steps`, and
    #     the new validator REFUSES the old spelling -- so recovering a run's
    #     own config out of its snapshot and feeding it to today's binary dies
    #     at flag-parse time, after a full packaging round.
    #   * the parameter tree moves on. A new default that adds or renames a
    #     module makes the checkpoint unrestorable, which surfaces as a
    #     CheckpointMismatchError minutes into the job.
    #
    # The snapshot is immutable and already built, so reusing it is also
    # strictly cheaper. Deliberate code changes belong in a NEW experiment,
    # where the comparison is honest, not smuggled in through a resume.
    if [ -n "$resume_xid" ]; then
      local prior_stagedir
      prior_stagedir=$(python3 -c "import json,os
d={}
for f in ('~/.tpu_jobs.json','~/.tpu_jobs_legacy.json'):
    try:
        d.update(json.load(open(os.path.expanduser(f))))
    except Exception:
        pass
print(d.get('$resume_xid',{}).get('stagedir',''))" 2>/dev/null)
      if [ -z "$prior_stagedir" ]; then
        echo -e "\033[31m[resume] No stagedir recorded for XID $resume_xid.\033[0m"
        echo -e "\033[31m  Looked in ~/.tpu_jobs.json and ~/.tpu_jobs_legacy.json.\033[0m"
        echo -e "\033[31m  Refusing to package the current checkout: a resume must re-run the\033[0m"
        echo -e "\033[31m  original snapshot. Pass --stagedir=<path> if you know it.\033[0m"
        return 1
      fi
      stagedir="$prior_stagedir"
      abs_stagedir="/google/src/cloud/qiaos/EqR-jax/google3/${stagedir}"
      if [ ! -d "$abs_stagedir" ]; then
        echo -e "\033[31m[resume] Recorded stagedir is gone: $abs_stagedir\033[0m"
        echo -e "\033[31m  Cannot resume XID $resume_xid without the code it ran.\033[0m"
        return 1
      fi
      echo -e "\033[32m[resume] Re-using the ORIGINAL snapshot (not the working tree):\033[0m"
      echo -e "\033[32m  $abs_stagedir\033[0m"
      echo -e "\033[2m  Local edits are deliberately NOT packaged. Launch a new experiment for those.\033[0m"
      export TPU_STAGEDIR="$abs_stagedir"
      export TPU_LOGDIR="$logdir"
      cd "$abs_stagedir"
    else
    mkdir -p "$abs_stagedir"
    echo "Snapshotting source codebase to CitC stagedir: $abs_stagedir"
    rsync -aL --exclude={'bazel-*','.citc','.git','__pycache__','*.npy','*.npz','*.ckpt','*.pth','*.pt','*.safetensors','data','logs','wandb'} ./ "$abs_stagedir/"
    if [ ! -f "$abs_stagedir/config.sh" ] && [ -f "$HOME/work/tpu_cmd/config.sh" ]; then
      cp "$HOME/work/tpu_cmd/config.sh" "$abs_stagedir/"
    fi
    if [ ! -f "$abs_stagedir/xm_launcher.py" ] && [ -f "$HOME/work/tpu_cmd/xm_launcher.py" ]; then
      cp "$HOME/work/tpu_cmd/xm_launcher.py" "$abs_stagedir/"
    fi
    if [ -f "$abs_stagedir/config.sh" ]; then
      sed -i "s|export TARGET_LABEL=.*|export TARGET_LABEL=\"//${stagedir}:main\"|g" "$abs_stagedir/config.sh"
    fi
    
    export TPU_STAGEDIR="$abs_stagedir"
    export TPU_LOGDIR="$logdir"
    cd "$abs_stagedir"
    fi

    # Process multiple groups (e.g. 1,2)
    IFS=',' read -ra GROUP_ARRAY <<< "$group"
    if [ ${#GROUP_ARRAY[@]} -gt 1 ]; then
      echo "⚠️  Warning: you specified multiple groups. Because of an XManager limitation this submits several completely independent experiments (they may all start running at once!)."
    fi

    for g in "${GROUP_ARRAY[@]}"; do
      local alloc=$(get_alloc_by_group_id "$g")
      if [ -z "$alloc" ]; then
        echo "Error: Unknown group number: $g"
        cd "$orig_dir"
        return 1
      fi
      
      local xm_args=(
        "xmanager" "launch" "$HOME/work/tpu_cmd/xm_launcher.py" "--"
        "--tpu_type=${tpu_type}"
        "--xm_resource_alloc=${alloc}"
      )
      
      # Auto-inject PROD tier for vqfree-xm (g5) to prevent hanging in BATCH priorities
      if [ "$g" = "5" ] && [ -z "$tier" ]; then
        tier="PROD"
      fi

      # Pass xm_tier if available
      if [ -n "$tier" ]; then
        xm_args+=("--tier=${tier}")
      fi
      
      # Append remaining arguments (e.g. override --config)
      xm_args+=("${passthrough_args[@]}")

      # LOCALITY GUARD. A job whose compute lands on a different continent from
      # its checkpoint bucket re-reads state across an ocean on every restart.
      # XID 275793223 did exactly that -- compute ske/eu, bucket tul/na, a 4 MB
      # checkpoint taking 223 s -- and Borg killed it for being slow to start,
      # 26 times in a row, without training a single step. The failure is
      # invisible: the job reports `running` the whole time.
      #
      # Only checks an EXPLICIT --cell, because that is the only placement known
      # before submit; the allocator's own choice is caught after the fact by
      # `tpu check`'s REGION column.
      _check_locality "$user_cell" "${passthrough_args[*]}"

      echo "================================"
      echo "Submitting into group: $alloc"
      echo "Running: ${xm_args[*]}"
      local log_file="${logdir}/xm_launch.log"
      "${xm_args[@]}" 2>&1 | tee "$log_file"

      # THE LAUNCH IS NOT DONE UNTIL XMANAGER SAYS SO. `tee` makes `$?` the exit
      # status of tee, not of the launcher, so the only trustworthy evidence
      # that an experiment was created is this line in the log. Six launches on
      # 2026-08-04 got as far as a successful build and then died between
      # `experiment.package()` and `experiment.add(job)` -- no traceback, no
      # work unit, and (before the launcher was reordered) a half-written
      # registry entry that `tpu check` showed forever as "unknown ... No
      # WorkUnits". Cause: the launcher is a ~3.75 GB PAR executed off BinFS,
      # and `binfsd` panics on a ~6h cycle during cache eviction; the remount
      # SIGBUSes every process holding an mmap of /google/bin. Nothing in the
      # launcher can catch that, so the check belongs out here.
      local xid=$(grep -oP 'Launched experiment \K\d+' "$log_file" | head -n 1)

      if [ -z "$xid" ]; then
        echo -e "\033[31m[launch] No 'Launched experiment' line -- the launcher died before creating the experiment.\033[0m"
        if grep -qiE 'SIGBUS|Signal 7|bad local file header|FailureSignalHandler' "$log_file"; then
          echo -e "\033[33m  Signature matches the BinFS remount fault (binfsd restarts ~every 6h during cache eviction).\033[0m"
        fi
        # Drop any entry the launcher pre-registered before dying, so a corpse
        # never reaches the status board. Harmless once the launcher registers
        # last; kept because a staged snapshot may still hold the old order.
        python3 - "$log_file" <<'PYEOF'
import json, os, re, sys, fcntl
log = sys.argv[1]
try:
    text = open(log, errors="replace").read()
except OSError:
    sys.exit(0)
# The launcher prints the XID it reserved even when it dies later.
m = re.search(r'https?://xids?/(\d+)|experiment_id[\'":= ]+(\d+)', text)
xid = next((g for g in (m.groups() if m else ()) if g), None)
if not xid:
    sys.exit(0)
path = os.path.expanduser("~/.tpu_jobs.json")
if not os.path.exists(path):
    sys.exit(0)
with open(path, "r") as f:
    fcntl.flock(f, fcntl.LOCK_SH)
    try:
        data = json.load(f)
    except Exception:
        sys.exit(0)
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
entry = data.get(xid)
# Only remove a CORPSE: an entry with no tier/alloc/status is one the launcher
# pre-registered and never finished. Never touch a healthy entry.
if entry is not None and not any(k in entry for k in ("tier", "alloc", "status")):
    data.pop(xid, None)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(data, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"  [launch] removed the orphaned registry entry for XID {xid}")
PYEOF
        if [ "${_TPU_LAUNCH_RETRIED:-0}" != "1" ]; then
          echo -e "\033[33m[launch] Retrying once (a fresh process re-mmaps /google/bin).\033[0m"
          export _TPU_LAUNCH_RETRIED=1
          "${xm_args[@]}" 2>&1 | tee "$log_file"
          xid=$(grep -oP 'Launched experiment \K\d+' "$log_file" | head -n 1)
          unset _TPU_LAUNCH_RETRIED
          [ -n "$xid" ] && echo -e "\033[32m[launch] Retry succeeded: XID $xid\033[0m"
        fi
        if [ -z "$xid" ]; then
          echo -e "\033[31m[launch] Still no experiment after a retry. Nothing was submitted.\033[0m"
          echo -e "\033[2m  If the BinFS signature appeared, the cache is likely over its limit:\033[0m"
          echo -e "\033[2m    grep 'bigger than desired size' /usr/local/google/tmp/binfsd.INFO | tail -1\033[0m"
        fi
      fi
      if [ -n "$xid" ]; then
          python3 - "$xid" "${tpu_type}" "${tier}" "${alloc}" "${logdir}" "${stagedir}" "${log_file}" << 'EOF'
import json, os, re, sys, fcntl

xid, tpu_type, tier, alloc, logdir, stagedir, log_file = sys.argv[1:8]

mapping_file = os.path.expanduser("~/.tpu_jobs.json")
data = {}
if os.path.exists(mapping_file):
    try:
        with open(mapping_file, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        pass

entry = data.get(xid, {})
exp_name = entry.get("exp_name", "")
if not exp_name and os.path.exists(log_file):
    try:
        with open(log_file, "r") as lf:
            log_content = lf.read()
        m_note = re.search(r'--config\.wandb\.(?:notes|run_name)=(\S+)', log_content)
        if m_note:
            exp_name = m_note.group(1).strip()
        else:
            m_cfg = re.search(r"--config=['\"]?([^'\"]+)", log_content)
            if m_cfg:
                exp_name = m_cfg.group(1).strip()
            else:
                m_exp = re.search(r'Experiment name:\s*(.+)', log_content)
                if m_exp:
                    exp_name = m_exp.group(1).strip()
    except Exception:
        pass

entry.update({
    "tpu_type": tpu_type,
    "tier": tier,
    "alloc": alloc,
    "logdir": logdir,
    "stagedir": stagedir,
    "launch_log": log_file,
    "exp_name": exp_name or entry.get("exp_name", "-"),
    "status": entry.get("status", "SUBMITTED"),
    "error": entry.get("error", ""),
    "retry_count": entry.get("retry_count", 0)
})
data[xid] = entry

with open(mapping_file, "w") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    json.dump(data, f, indent=2)
    fcntl.flock(f, fcntl.LOCK_UN)
EOF
          echo "Successfully registered XID $xid in ~/.tpu_jobs.json"

          # Cap this XID's price. Per-XID scope: does not affect teammates and
          # survives the periodic group-wide push (SCU > XID > MDB).
          if [ "$no_limit_order" != "1" ]; then
            _tpu_set_limit_order "$xid" "$tpu_type" "$tier" "$lo_price"
          else
            echo -e "\033[2m[limit order] Skipped (--no-limit-order). Job runs at market price.\033[0m"
          fi
      fi
    done
    cd "$orig_dir"

  elif [[ "$1" == "check" || "$1" == "c" ]]; then
    shift
    python3 - "$@" << 'EOF'
import argparse, json, os, re, sys, fcntl

def remove_ansi(text):
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)

def _trunc(s, cap):
    s = "" if s is None else str(s)
    if cap is None or len(s) <= cap:
        return s
    return s[:cap - 1] + "…"

def _c(s, color):
    if not color:
        return s
    return f"{color}{s}\033[0m"

STATUS_COLOR = {
    "RUNNING": "\033[32m", "running": "\033[32m", "active": "\033[32m", "ACTIVE": "\033[32m",
    "STARTING": "\033[34m", "starting": "\033[34m", "SUBMITTED": "\033[36m", "submitted": "\033[36m",
    "PENDING": "\033[33m", "pending": "\033[33m", "queued": "\033[33m",
    "FINISHED": "\033[32m", "finished": "\033[32m", "COMPLETED": "\033[35m", "completed": "\033[35m",
    "STOPPED": "\033[35m", "stopped": "\033[35m", "FAILED": "\033[31m", "failed": "\033[31m",
    "KILLED": "\033[31m", "killed": "\033[31m", "CANCELLED": "\033[2m", "unknown": "\033[2m",
}

BOLD = "\033[1m"
DIM = "\033[2m"

def print_table(headers, rows, caps, status_idx=None, last_dim=True, tails=None):
    """`tails` maps a row's index to extra dim lines printed beneath it.

    A running job gets its log tail this way: the status word only says Borg is
    happy, so the line the job actually last printed is what tells you whether
    training is moving. Indented under the NAME column and dimmed, so the table
    still scans vertically.
    """
    disp = [[_trunc(c, caps[i]) for i, c in enumerate(r)] for r in rows]
    n = len(headers)
    width = [len(h) for h in headers]
    for r in disp:
        for i in range(n):
            width[i] = max(width[i], len(r[i]))
    print(_c("  " + "  ".join(h.ljust(width[i]) for i, h in enumerate(headers)), DIM))
    # Line the continuation up with NAME (col 2), which is where the eye already is.
    tail_indent = 2 + sum(width[i] + 2 for i in range(min(2, n)))
    for idx, r in enumerate(disp):
        parts = []
        for i, cell in enumerate(r):
            last = i == n - 1
            padded = cell if last else cell.ljust(width[i])
            if i == status_idx:
                padded = _c(padded, STATUS_COLOR.get(cell, ""))
            elif last and last_dim:
                padded = _c(padded, DIM)
            parts.append(padded)
        print("  " + "  ".join(parts))
        for extra in (tails or {}).get(idx, []):
            print(_c(" " * tail_indent + "│ " + extra, DIM))

def _hdr(title, n):
    bar = _c(f"━━ {title} ", BOLD)
    return f"\n{bar}" + _c(f"({n})", DIM)

def main():
    parser = argparse.ArgumentParser(description="Rich status view for TPU jobs (like infra check).")
    parser.add_argument("-a", "--all", action="store_true", help="Show all completed/failed jobs.")
    # -l is the alias people reach for ("long"), -f the original. Same switch:
    # an experiment name is chosen to be read, and 30 characters of it plus an
    # ellipsis routinely hides the very thing that distinguishes two runs --
    # "...50k scaleup" vs "...50k scaleup lr2x" truncate identically.
    parser.add_argument("-f", "-l", "--full", "--long", action="store_true",
                        dest="full",
                        help="Do not truncate the NAME / GROUP columns.")
    parser.add_argument("-d", "--done", type=int, default=10, help="Number of done jobs to display.")
    args, _ = parser.parse_known_args()

    tpu_jobs = {}
    mapping_file = os.path.expanduser("~/.tpu_jobs.json")
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                tpu_jobs = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass

    cached_status = {}
    cache_file = os.path.expanduser("~/.tpu_check_cache.txt")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                lines = f.readlines()
            last_xid = None
            for line in lines:
                clean_line = remove_ansi(line).strip()
                # Only require a LEADING box char. Requiring a trailing one too
                # meant that a table wider than the render width -- which clips
                # the right edge -- matched zero rows, and every job silently
                # fell back to "SUBMITTED". A parser for a display format must
                # degrade, not go blank.
                if clean_line.startswith("│"):
                    cells = clean_line.split("│")
                    parts = [p.strip() for p in (cells[1:-1] if clean_line.endswith("│") else cells[1:])]
                    if len(parts) >= 3:
                        xid = parts[0]
                        if xid.isdigit():
                            # Layout: ID | STATUS | NAME | RESUME | STEP | (DETAILS|WHY)
                            # Index the tail, not a fixed column: the reason is
                            # always last, and hard-coding parts[3] silently
                            # returned the wrong field when columns were added.
                            status = parts[1]
                            name = parts[2]
                            resume = parts[3] if len(parts) >= 6 else ""
                            step = parts[4] if len(parts) >= 6 else ""
                            why = parts[-1] if len(parts) >= 4 else ""
                            cached_status[xid] = {"status": status, "name": name,
                                                  "why": why, "resume": resume,
                                                  "step": step, "tail": []}
                            last_xid = xid
                        elif not xid and last_xid:
                            # CONTINUATION ROW. infra_check prints a running
                            # job's log tail as extra rows whose ID column is
                            # blank (see infra_check.py::_log_tail). Requiring a
                            # numeric ID dropped every one of them, which is why
                            # the tail was invisible here while being present in
                            # the cache file. Attribute the row to the job whose
                            # header we last saw.
                            text = next((p for p in parts[2:] if p), "")
                            text = text.lstrip("│ ").strip()
                            if text:
                                cached_status[last_xid].setdefault("tail", []).append(text)
        except Exception:
            pass

    all_xids = sorted(list(set(tpu_jobs.keys()) | set(cached_status.keys())), key=lambda x: int(x) if x.isdigit() else x, reverse=True)

    active_rows, pending_rows, done_rows = [], [], []
    # row index in active_rows -> its log-tail continuation lines
    active_tails = {}

    dirty_jobs = False
    for xid in all_xids:
        job_info = tpu_jobs.get(xid, {})
        cache_info = cached_status.get(xid, {})

        status = cache_info.get("status") or job_info.get("status") or "SUBMITTED"
        name = job_info.get("exp_name") or cache_info.get("name") or ""
        if not name or name == "-":
            launch_log = job_info.get("launch_log", "")
            if launch_log and os.path.exists(launch_log):
                try:
                    with open(launch_log, "r") as lf:
                        content = lf.read()
                    m_note = re.search(r"--config\.wandb\.(?:notes|run_name)=(\S+)", content)
                    if m_note:
                        name = m_note.group(1).strip()
                    else:
                        m_cfg = re.search(r"--config=['\"]?([^'\"]+)", content)
                        if m_cfg:
                            name = m_cfg.group(1).strip()
                        else:
                            m_exp = re.search(r"Experiment name:\s*(.+)", content)
                            if m_exp:
                                name = m_exp.group(1).strip()
                except Exception:
                    pass
        if not name:
            name = "-"
        elif xid in tpu_jobs and not tpu_jobs[xid].get("exp_name"):
            tpu_jobs[xid]["exp_name"] = name
            dirty_jobs = True

        tpu_type = job_info.get("tpu_type") or "-"
        tier = job_info.get("tier") or "-"
        # LINT.IfChange(group_map) — keep in sync with get_alloc_by_group_id above.
        ALLOC_MAP = {
            "group:deepmind-dynamic/gdm-resources-prod-shared-users-dynamic": "g1",
            "group:deepmind-dynamic/gdm-viscam-goflow-dynamic": "g2",
            "group:deepmind-dynamic/gdm-viscam-interns-dynamic": "g3",
            "group:deepmind-dynamic/viscam-interns": "g4",
            "group:deepmind-dynamic/vqfree-xm": "g5",
            "group:dm/deepmind-large-scale-workshop": "g6",
            "group:dm/dm-resources-prod-shared": "g7",
            "group:gdm-aux/brain-vasp-shared-user-xm": "g8",
            "group:deepmind-dynamic/fr-dna-grand-challenge-team-resource": "g9",
        }
        # LINT.ThenChange(above)
        alloc = job_info.get("alloc") or "-"
        if alloc in ALLOC_MAP:
            group_str = ALLOC_MAP[alloc]
        elif alloc.isdigit():
            group_str = f"g{alloc}"
        else:
            group_str = alloc.replace("group:", "")
        why = cache_info.get("why") or job_info.get("error") or ""
        if not why or why == "unknown reason":
            if "failed" in status.lower():
                why = "Rejected by Allocator/Borg"
            else:
                why = "-"

        retry_count = job_info.get("retry_count", 5)
        retry_timer_start = job_info.get("retry_timer_start", 0)
        
        if retry_count > 0 and retry_count < 5:
            name = f"{name} (Retry {retry_count})"
            
        if "failed" in status.lower() and tier == "PROD" and "Rejected by Allocator/Borg" in why and retry_count < 5:
            import time
            rem = 300 - int(time.time() - retry_timer_start)
            if rem < 0: rem = 0
            why = f"Retrying in {rem//60}m{rem%60}s ({retry_count}/5)"
            status = "PENDING"

        st_lower = status.lower()
        why_lower = why.lower()

        # A preempted job is NOT pending. Borg counts the torn-down gang as a
        # task FAILURE and (with max_task_failures=0) declares the job dead --
        # nothing will ever re-queue it. Only rewrite to PENDING when the job
        # is genuinely still alive; otherwise keep the terminal status and just
        # say WHY it died. Previously any "preempt" substring won over the real
        # failed state, so dead jobs showed as PENDING forever.

        resume_s = cache_info.get("resume") or "-"
        step_s = cache_info.get("step") or "-"

        # Per-section columns, following unified_infra's `infra check`: each
        # section answers a different question, so a shared header wastes width
        # on cells that are structurally "-".
        #   active  -> "is it progressing?"  WU replaces WHY, whose only value
        #              for a live job was the work-unit count it already carries.
        #   pending -> "why is it not running yet?"  WHY is the whole point here
        #              (GQM_RESOURCE_DEFICIT_INFO, "Retrying in 3m20s (2/5)"),
        #              while RESUME/STEP are always "-" before a job starts.
        #   done    -> "how did it end?"  WHY again, but RESUME is meaningless
        #              once terminal.
        if st_lower in ["running", "active", "starting", "submitted", "staging"]:
            # `why` for a live job is the daemon's "<n> active" work-unit line;
            # surface just the count, and fall back to the raw text if the
            # daemon ever reports something else (an early-startup message).
            m_wu = re.match(r"^\s*(\d+)\s+active\s*$", str(why or ""))
            wu = m_wu.group(1) if m_wu else ("-" if why in ("", "-", None) else why)
            tail = cache_info.get("tail") or []
            if tail:
                active_tails[len(active_rows)] = tail
            active_rows.append([xid, status, name, tpu_type, tier, group_str,
                                resume_s, step_s, wu])
        elif st_lower in ["pending", "queued"]:
            pending_rows.append([xid, status, name, tpu_type, tier, group_str, why])
        else:
            done_rows.append([xid, status, name, tpu_type, tier, group_str,
                              step_s, why])

    if dirty_jobs:
        try:
            with open(mapping_file, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(tpu_jobs, f, indent=2)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass

    user = os.environ.get("USER", "qiaos")
    print(_c("tpu check", BOLD) + _c(f"  ({user})", DIM))

    name_cap = None if args.full else 30
    group_cap = None if args.full else 20

    # Each section has its own header/caps pair; see the row-building comment.
    # WU = live work units. A running job's WHY was always just "<n> active",
    # which is a count, not a reason -- as a column it is one char wide instead
    # of eating the line.
    active_headers = ["XID", "STATUS", "NAME", "TPU", "TIER", "GROUP", "RESUME", "STEP", "WU"]
    active_caps = [None, None, name_cap, None, None, group_cap, None, None, None]

    pending_headers = ["XID", "STATUS", "NAME", "TPU", "TIER", "GROUP", "WHY"]
    pending_caps = [None, None, name_cap, None, None, group_cap, None]

    done_headers = ["XID", "STATUS", "NAME", "TPU", "TIER", "GROUP", "STEP", "WHY"]
    done_caps = [None, None, name_cap, None, None, group_cap, None, None]

    print(_hdr("active", len(active_rows)))
    if active_rows:
        # last_dim=False: WU is a live datum, not a trailing note, so it should
        # not be dimmed the way a WHY string is.
        print_table(active_headers, active_rows, active_caps, status_idx=1,
                    last_dim=False, tails=active_tails)
    else:
        print(_c("  (none)", DIM))

    print(_hdr("pending", len(pending_rows)))
    if pending_rows:
        print_table(pending_headers, pending_rows, pending_caps, status_idx=1)
    else:
        print(_c("  (none)", DIM))

    done_limit = len(done_rows) if args.all else args.done
    done_disp = done_rows[:done_limit]
    print(_hdr("recent done", len(done_rows)))
    if done_disp:
        print_table(done_headers, done_disp, done_caps, status_idx=1)
    else:
        print(_c("  (none)", DIM))

if __name__ == "__main__":
    main()
EOF

  elif [[ "$1" == "cancel" || "$1" == "stop" ]]; then
    shift
    local dry_run=0
    local xids=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --dry-run|--dry_run)
          dry_run=1
          shift
          ;;
        -*)
          echo "Error: Unsupported option '$1'"
          echo "Usage: tpu cancel <xid> [xid...] [--dry-run]"
          return 1
          ;;
        *)
          xids+=("$1")
          shift
          ;;
      esac
    done
    if [ ${#xids[@]} -eq 0 ]; then
      echo "Usage: tpu cancel <xid> [xid...] [--dry-run]"
      echo "  Stops the XManager experiment(s), i.e. all work units and their Borg jobs,"
      echo "  then marks them CANCELLED in ~/.tpu_jobs.json so 'tpu check' reflects it"
      echo "  immediately and the PROD auto-retry daemon cannot resubmit them."
      echo "  --dry-run shows what xmanager would stop without stopping anything."
      return 1
    fi
    local x
    for x in "${xids[@]}"; do
      if ! [[ "$x" =~ ^[0-9]+$ ]]; then
        echo -e "\033[31mError: '$x' is not a numeric XID.\033[0m"
        return 1
      fi
    done
    local ids
    ids=$(IFS=,; echo "${xids[*]}")
    local stop_args=("--experiment_id=${ids}" "--skip_confirmation")
    if [ "$dry_run" = "1" ]; then
      stop_args+=("--dry_run")
      echo -e "\033[36m[tpu cancel] DRY RUN on XID(s) ${ids}...\033[0m"
    else
      echo -e "\033[36m[tpu cancel] Stopping XID(s) ${ids} via 'xmanager stop'...\033[0m"
    fi
    local cancel_log="/tmp/tpu_cancel_$$.log"
    xmanager stop "${stop_args[@]}" > "$cancel_log" 2>&1
    local stop_status=$?
    # The CLI prints ~20 lines of build/absl preamble before the result table.
    grep -vE "^(INFO:absl|WARNING: Logging|W[0-9]{4} |Built |Build |Currently running)" "$cancel_log"
    if [ "$stop_status" -ne 0 ]; then
      echo -e "\033[31m[tpu cancel] xmanager stop exited with ${stop_status}; registry left untouched.\033[0m"
      echo -e "\033[2m  Full output: $cancel_log\033[0m"
      return "$stop_status"
    fi
    if [ "$dry_run" = "1" ]; then
      return 0
    fi
    python3 - "${xids[@]}" << 'EOF'
import json, os, sys, fcntl, time

xids = sys.argv[1:]
mapping_file = os.path.expanduser("~/.tpu_jobs.json")
if not os.path.exists(mapping_file):
    sys.exit(0)
try:
    with open(mapping_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
        except ValueError:
            data = {}
        changed = False
        for xid in xids:
            entry = data.get(xid)
            if entry is None:
                continue
            entry["status"] = "CANCELLED"
            entry["error"] = ""
            entry["cancelled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            # The daemon resubmits PROD jobs whose status is FAILED with
            # retry_count < 5; pinning the counter makes a cancelled job
            # unresurrectable even if its cached status later reads failed.
            entry["retry_count"] = 5
            changed = True
        if changed:
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)
except Exception as e:  # never fail the cancel over bookkeeping
    print(f"Warning: could not update {mapping_file}: {e}")
EOF
    echo -e "\033[32m[tpu cancel] Done. Marked ${ids} CANCELLED in ~/.tpu_jobs.json.\033[0m"
    echo -e "\033[2m  'tpu check' shows the live XManager state after the next daemon cycle (~60s).\033[0m"

  elif [[ "$1" == "quota" ]]; then
    local FLAG="$2"
    local ARG="$3"
    local CACHE_DIR="$HOME/.tpu_quota_cache_dir"
    
    # Check defaults & freshness
    if [ -f "$CACHE_DIR/default.txt" ]; then
        local AGE=$(($(date +%s) - $(stat -c %Y "$CACHE_DIR/default.txt")))
        if [ "$AGE" -gt 180 ]; then
            echo -e "\033[31m[tpu quota] 🚨 ALERT: quota data is stale (last updated ${AGE}s ago, over the 3-minute limit)!\033[0m"
            echo -e "\033[33mLikely cause: LOAS / gcert credentials expired, or the background poller died.\033[0m"
            echo -e "\033[36m[auto-recovery] Restarting the tpu-daemon background tmux session...\033[0m"
            
            tmux kill-session -t tpu-daemon 2>/dev/null || true
            tmux new-session -d -s tpu-daemon 'bash -c "while true; do /usr/local/google/home/qiaos/work/tpu_check_daemon.sh; echo Daemon crashed, restarting in 5s...; sleep 5; done"'
            
            echo -e "\033[33m👉 Make sure gcert is valid (run 'gcert' to re-authenticate if it expired), then re-run 'tpu quota'.\033[0m"
            return 1
        fi
        echo -e "\033[36m[tpu quota] ⚡ Zero-latency offline read (background refresh ${AGE}s ago)\033[0m"
    else
        echo -e "\033[31m[tpu quota] 🚨 Cache not initialized! Starting the tpu-daemon background tmux poller...\033[0m"
        tmux kill-session -t tpu-daemon 2>/dev/null || true
        tmux new-session -d -s tpu-daemon 'bash -c "while true; do /usr/local/google/home/qiaos/work/tpu_check_daemon.sh; echo Daemon crashed, restarting in 5s...; sleep 5; done"'
        return 1
    fi
    
    if [[ "$FLAG" == "-l" ]]; then
        cat "$CACHE_DIR/list.txt"
    elif [[ "$FLAG" == "-g" ]]; then
        if [[ -n "$ARG" ]] && [[ -f "$CACHE_DIR/g${ARG}.txt" ]]; then
            cat "$CACHE_DIR/g${ARG}.txt"
        else
            echo -e "\033[31mError: no data found for Group $ARG. Run 'tpu quota' to see the available group numbers.\033[0m"
        fi
    else
        cat "$CACHE_DIR/default.txt"
    fi

  elif [[ "$1" == "money" || "$1" == "m" || "$1" == "price" ]]; then
    local CACHE_DIR="$HOME/.tpu_quota_cache_dir"
    if [ -f "$CACHE_DIR/money.txt" ]; then
        local AGE=$(($(date +%s) - $(stat -c %Y "$CACHE_DIR/money.txt")))
        if [ "$AGE" -gt 180 ]; then
            echo -e "\033[31m[tpu money] 🚨 ALERT: bidding power / price data is stale (last updated ${AGE}s ago, over the 3-minute limit)!\033[0m"
            echo -e "\033[33mLikely cause: LOAS / gcert credentials expired, or the background poller died.\033[0m"
            echo -e "\033[36m[auto-recovery] Restarting the tpu-daemon background tmux session...\033[0m"
            
            tmux kill-session -t tpu-daemon 2>/dev/null || true
            tmux new-session -d -s tpu-daemon 'bash -c "while true; do /usr/local/google/home/qiaos/work/tpu_check_daemon.sh; echo Daemon crashed, restarting in 5s...; sleep 5; done"'
            
            echo -e "\033[33m👉 Make sure gcert is valid (run 'gcert' to re-authenticate if it expired), then re-run 'tpu money'.\033[0m"
            return 1
        fi
        echo -e "\033[36m[tpu money] ⚡ Zero-latency offline read (background refresh ${AGE}s ago)\033[0m"
        cat "$CACHE_DIR/money.txt"
    else
        echo -e "\033[31m[tpu money] 🚨 Cache not initialized! Starting the tpu-daemon background tmux poller...\033[0m"
        tmux kill-session -t tpu-daemon 2>/dev/null || true
        tmux new-session -d -s tpu-daemon 'bash -c "while true; do /usr/local/google/home/qiaos/work/tpu_check_daemon.sh; echo Daemon crashed, restarting in 5s...; sleep 5; done"'
        return 1
    fi

  elif [[ "$1" == "preflight" || "$1" == "pf" ]]; then
    shift
    if [ ! -x "$_PREFLIGHT_CLI_BIN" ]; then
      echo -e "\033[31mPreflight binary not built. Run:\033[0m"
      echo "  (cd /google/src/cloud/qiaos/xm_test/google3 && blaze build experimental/users/qiaos/tpu_utils/preflight:preflight_cli)"
      return 1
    fi
    "$_PREFLIGHT_CLI_BIN" "$@"

  elif [[ "$1" == "route" || "$1" == "router" || "$1" == "r" ]]; then
    shift
    if [ ! -x "$_ROUTER_CLI_BIN" ]; then
      echo -e "\033[31mRouter binary not built. Run:\033[0m"
      echo "  (cd /google/src/cloud/qiaos/xm_test/google3 && blaze build experimental/users/qiaos/tpu_utils/preflight:router_cli)"
      return 1
    fi
    "$_ROUTER_CLI_BIN" "$@"

  elif [[ "$1" == "clear" ]]; then
    # Archive (NOT delete) jobs off the status board. The implementation lives in
    # the google3 half -- `infra_check.py::main` branches on argv[1]=='clear' --
    # and this branch is the only thing that reaches it. Without it, `tpu clear`
    # fell through to the final `command tpu`, which does not exist, so the
    # documented command died with "tpu: command not found".
    shift
    if [ ! -x "$_INFRA_CHECK_BIN" ]; then
      echo -e "\033[31minfra_check binary not built. Run:\033[0m"
      echo "  (cd /google/src/cloud/qiaos/xm_test/google3 && blaze build experimental/users/qiaos/tpu_utils:infra_check)"
      return 1
    fi
    "$_INFRA_CHECK_BIN" clear "$@"
    # `tpu check` renders from ~/.tpu_check_cache.txt, which the daemon rewrites
    # on its own ~60s cycle, so the board does not change the instant this returns.
    echo -e "\033[2m[tpu clear] Entries archived to ~/.tpu_jobs_legacy.json (never deleted).\033[0m"
    echo -e "\033[2m  Allow one daemon cycle (~60s) for 'tpu check' to drop them from the board.\033[0m"

  elif [[ "$1" == "gc" ]]; then
    # Prune checkpoints nothing will read again. `save_checkpoint` uses orbax's
    # StandardCheckpointer, which has no `max_to_keep` -- so before this existed
    # NOTHING ever deleted a checkpoint, and 1850 of them across one cell put a
    # personal 500 GiB CNS quota over its ceiling, poisoning every later write.
    # Dry run by default; --go deletes. See scripts/ckpt_gc.py for the policy.
    shift
    python3 "$(dirname "${BASH_SOURCE[0]}")/scripts/ckpt_gc.py" "$@"

  elif [[ "$1" == "monitor" || "$1" == "watch" ]]; then
    shift
    while true; do
      clear
      echo -e "\033[2m[tpu monitor] Live mode: refreshing every 5s. Press Ctrl-C to exit...\033[0m"
      echo ""
      tpu check "$@"
      sleep 5
    done

  else
    command tpu "$@"
  fi
}
