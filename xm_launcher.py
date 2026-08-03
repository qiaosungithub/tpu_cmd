r"""Launcher for running EqR-jax on TPUs using XManager.

Usage example:
xmanager launch xm_launch.py -- \
  --xm_resource_alloc="group:gdm-aux/brain-vasp-shared-user-xm" \
  --tpu_type="v4-64,v5p-32" \
  --config="remote_run" \
  --workdir="~/logs/eqr-run"
"""
from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib import framework_defaults
from xmanager.contrib.internal import xm_jax

_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'eqr-jax', 'Name of the experiment.', short_name='n'
)
_CONFIG = flags.DEFINE_string(
    'config', 'remote_run', 'The configs/load_config.py:<mode> to run.'
)

_BUCKET = flags.DEFINE_string(
    'bucket', '/cns/yutulpz-d/home/qiaos/eqr_data',
    'Durable root for checkpoints and mirrored logs. Defaults to CNS because a '
    'Borg job runs as <user>@prod.google.com, a different IAM principal from '
    'the <user>@google.com that owns our GCS bucket -- every gs:// write from a '
    'TPU worker fails with ACCESS_DENIED. Pass a gs:// path only if that bucket '
    'grants access to the prod identity.'
)
# Checkpoints are written from the TPU workers, so the bucket has to be near
# THEM, not near wherever the default happens to point. Getting this wrong is
# not a mild slowdown: XID 275990419 ran on `yuskedq` (metro ske, continent EU)
# while writing to `yutulpz` (metro tul, NA), and orbax reported 10 MiB/s, ~10s
# of BLOCKED TPU per save plus 33-56s of background flush. Its duty cycle fell
# to 0.082, below the 0.20 WIM pruning threshold, and the job was deleted.
#
# Map each compute cell to a bucket in the same metro. An unlisted cell keeps
# the default, and an explicit --bucket always wins.
_CELL_BUCKETS = {
    'yuskedq': '/cns/yuskedq-d/home/qiaos/eqr_data',
}


def _read_legacy_mapping():
    """Archived job registry (`tpu clear` moves entries here). {} if absent.

    `os`/`json` are imported locally to match this module's existing style --
    they are function-local everywhere else here, not module-level.
    """
    import os
    import json
    try:
        with open(os.path.expanduser("~/.tpu_jobs_legacy.json"), "r") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001 - the archive is optional
        return {}


def _local_bucket() -> str:
    """The durable root nearest the cell this job will run in."""
    # An explicitly passed --bucket is authoritative.
    if _BUCKET.present:
        return _BUCKET.value
    cell = (_CELL.value or '').strip()
    for name, bucket in _CELL_BUCKETS.items():
        if name == cell:
            print(f"[locality] cell={cell}: using co-located bucket {bucket}")
            return bucket
    return _BUCKET.value


_WORKDIR = flags.DEFINE_string(
    'workdir', '', 'Working directory (e.g. ~/logs/...) '
)
_TPU_TYPE = flags.DEFINE_string(
    'tpu_type', 'v4-8', 'Comma-separated TPU specs e.g. v4-64,v5p-32'
)
_RESUME_XID = flags.DEFINE_integer(
    'resume_xid', 0, 'If set, appends job to the given existing XManager experiment ID instead of creating a new one.'
)
# Failure budget shaped after //experimental/.../mesh_diffusion launch_lib.py:
# unlimited total failures, but a tight per-task limit that decays over time.
# The asymmetry is the point -- a long run should survive any number of
# unrelated preemptions, while a task that keeps dying is a real bug and should
# be declared dead quickly rather than retried forever.
_MAX_TASK_FAILURES = flags.DEFINE_integer(
    'borg_max_task_failures', -1,
    'Total task failures tolerated across all tasks before the job is aborted. '
    '-1 means unlimited. Borg default is 0, which kills the whole experiment '
    'the first time a preempted gang tears down. Applies to PROD and BATCH: '
    'PROD is also preemptible via equal-priority slice defragmentation.'
)
_MAX_PER_TASK_FAILURES = flags.DEFINE_integer(
    'borg_max_per_task_failures', 1,
    'Failures tolerated per individual task before that task is declared dead. '
    'Combined with the credit period below this reads as "recover from at most '
    'one failure per task every N seconds".'
)
_FAILURE_CREDIT_PERIOD = flags.DEFINE_integer(
    'borg_failure_credit_period', 7200,
    'Every N seconds Borg decrements each live task\'s failure count, so a run '
    'is not killed by slow attrition of unrelated one-off failures.'
)
# XManager derives the Borg job name from the packaged target ('main'); naming
# the job explicitly keeps that in sync with the BCL token built below.
_JOB_NAME = 'main'

# Borg applies its own default RAM when the requirements block names none, and
# that default is sized for a small server, not for a job holding several
# copy buffers at once. There was no way to ask for more: the launcher builds
# JobRequirements from the accelerator string alone, and `--tpu_type=cpu=1,...`
# is parsed as a SECOND accelerator (one executor per comma-separated entry),
# not as a second resource. Hence this flag. 0 keeps today's behaviour exactly
# -- no `ram` is emitted and Borg's default applies -- so no existing job moves.
_RAM_GIB = flags.DEFINE_float(
    'ram_gib', 0.0,
    'Per-task RAM requirement in GiB. 0 (default) omits it and lets Borg '
    'choose, which is what every job did before this flag existed. Distinct '
    'from --tmp_ram_fs_gib: that sizes the RAM DISK backing /tmp, this sizes '
    'the memory the process may allocate.'
)
_TMP_RAM_FS_GIB = flags.DEFINE_integer(
    'tmp_ram_fs_gib', 16,
    'Size of the per-task RAM disk backing /tmp, in GiB. Must exceed whatever '
    'the job stages locally (dataset copies, scratch).'
)
_LOAD_FROM = flags.DEFINE_string(
    'load_from', '',
    'Checkpoint directory to evaluate (eval_only) or warm-start from. Accepts '
    'a gs:// path or a local path. Exported to the job as $LOAD_FROM rather '
    'than as a --config flag; see unified_infra infra/runjob.py:137. A value '
    'here overrides any load_from seeded in the yaml config.'
)
_WANDB_RESUME_ID = flags.DEFINE_string(
    'wandb_resume_id', '',
    'Run id to resume experiment tracking under. Exported as $WANDB_RESUME_ID.'
)
_CELL = flags.DEFINE_string(
    'cell', '',
    'Borg cell to pin the job to (e.g. nz, pa, go), or "viglobal" to let XBorg '
    'choose. GQM clears prices PER CELL, so pinning to a cheap cell is often '
    'the difference between running and sitting in the '
    'TRIGGERED_LIMIT_ORDER bucket. Empty = let the allocator decide.'
)

def main(argv) -> None:
    exp_name = _EXP_NAME.value
    # --- Auto-Load WandB name or fallbacks ---
    try:
        from configs import load_config
        cfg = load_config.get_config(_CONFIG.value)
        if getattr(cfg, 'wandb', None):
            if getattr(cfg.wandb, 'notes', None):
                exp_name = cfg.wandb.notes
            elif getattr(cfg.wandb, 'run_name', None):
                exp_name = cfg.wandb.run_name
    except Exception:
        pass
    for arg in argv[1:]:
        if arg.startswith('--config.wandb.notes='):
            exp_name = arg.split('=', 1)[1]
        elif arg.startswith('--config.wandb.run_name='):
            exp_name = arg.split('=', 1)[1]
            
    experiment_context = xm_abc.get_experiment(experiment_id=_RESUME_XID.value) if _RESUME_XID.value else xm_abc.create_experiment(experiment_title=exp_name)
    with experiment_context as experiment:
        
        # --- Codebase Specifics via config.sh ---
        project_name = "unified_project"
        package_mode = "python" # 'python' or 'bazel'
        target_label = "."
        
        pkg_path = target_label.lstrip('/').split(':')[0]
        
        try:
            import os
            config_path = "config.sh"
            if not os.path.exists(config_path):
                config_path = f"{pkg_path}/config.sh"
            with open(config_path, "r") as f:
                for line in f:
                    if line.startswith("export PROJECT_NAME="):
                        project_name = line.split("=")[1].strip().strip('"').strip("'")
                    elif line.startswith("export PACKAGE_MODE="):
                        package_mode = line.split("=")[1].strip().strip('"').strip("'")
                    elif line.startswith("export TARGET_LABEL="):
                        target_label = line.split("=")[1].strip().strip('"').strip("'")
                        pkg_path = target_label.lstrip('/').split(':')[0]
        except Exception:
            pass

        executors = []
        is_tpu_job = False
        tpu_types = [t.strip() for t in _TPU_TYPE.value.split(',')]
        # Borg ScalarResource codenames. Source of truth:
        #   //depot/google3/third_party/py/xmanager/xm/resources.py (ResourceType)
        #   //depot/google3/borg/common/scalar_resource.proto
        # NOTE: GHOSTFISHLITE (101) is v7, NOT v5e -- v5e is VIPERLITE_POD (62).
        FISH_MAP = {
            "v4": "pufferfish",       # 34
            "v4lite": "dragonfish",   # 16
            "v5e": "viperlite_pod",   # 62
            "v5p": "viperfish",       # 59
            "v6e": "ghostlite_pod",   # 63
            "v6p": "ghostfish",       # 92
            "v7": "ghostfishlite",    # 101
        }
        
        # LINT.IfChange(group_map) — keep in sync with tpu_wrapper.sh & group_utils.py.
        _GROUP_MAP = {
            '1': 'group:deepmind-dynamic/gdm-resources-prod-shared-users-dynamic',
            '2': 'group:deepmind-dynamic/gdm-viscam-goflow-dynamic',
            '3': 'group:deepmind-dynamic/gdm-viscam-interns-dynamic',
            '4': 'group:deepmind-dynamic/viscam-interns',
            '5': 'group:deepmind-dynamic/vqfree-xm',
            '6': 'group:dm/deepmind-large-scale-workshop',
            '7': 'group:dm/dm-resources-prod-shared',
            '8': 'group:gdm-aux/brain-vasp-shared-user-xm',
            '9': 'group:deepmind-dynamic/fr-dna-grand-challenge-team-resource',
        }
        # LINT.ThenChange(//depot/google3/experimental/users/qiaos/tpu_utils/group_utils.py)
        alloc_str = None
        for arg in argv[1:]:
            if arg.startswith('--xm_resource_alloc='):
                alloc_str = arg.split('=', 1)[1]
            elif arg.startswith('--group='):
                group_val = arg.split('=', 1)[1]
                alloc_str = _GROUP_MAP.get(group_val, f'group:{group_val}')

        for tpu_str in tpu_types:
            if '-' in tpu_str and not '=' in tpu_str:
                arch, cores = tpu_str.split('-', 1)
                # Official Google3 TPU topology mappings (from learning/performance/ace/search_space_utils.py)
                TORUS_3D_MAP = {
                    "1": "1x1x1",
                    "2": "1x2x1",
                    "4": "2x2x1",
                    "8": "2x2x2",
                    "16": "2x2x4",
                    "32": "2x4x4",
                    "64": "4x4x4",
                    "128": "4x4x8",
                    "256": "4x8x8",
                    "512": "4x8x16",
                }
                TORUS_2D_MAP = {
                    "1": "1x1",
                    "4": "2x2",
                    "8": "2x4",
                    "16": "4x4",
                    "32": "4x8",
                    "64": "8x8",
                    "128": "8x16",
                    "256": "16x16",
                }
                arch_lower = arch.lower()
                
                num_cores = int(cores) if cores.isdigit() else 0
                is_prod_pool = alloc_str and 'deepmind-dynamic' in alloc_str
                
                # v7 (ghostfishlite) is a 3-D torus with 4 chips/host, the same
                # geometry as v6p: ghostfishlite.gcl and ghostfish.gcl declare an
                # identical static sub-cube list and dynamic-slice rule, differing
                # only in the locus name. So it takes the 3-D branch below.
                if arch_lower in ["v4", "pufferfish", "v5p", "viperfish", "v6p", "ghostfish",
                                  "v7", "ghostfishlite"]:
                    min_allowed = 16 if is_prod_pool else 8
                    if num_cores > 0 and num_cores < min_allowed:
                        raise ValueError(f"[BLOCKED] In {alloc_str or 'the current resource pool'}, the minimum allowed slice for {arch} is {min_allowed} chips, but you requested {num_cores}. To avoid an instant allocator rejection, request at least {arch}-{min_allowed}.")
                    if cores in TORUS_3D_MAP:
                        cores = TORUS_3D_MAP[cores]
                elif arch_lower in ["v5e", "v6e", "viperlite_pod", "ghostlite_pod"]:
                    min_allowed = 16 if is_prod_pool else 4
                    if num_cores > 0 and num_cores < min_allowed:
                        raise ValueError(f"[BLOCKED] In {alloc_str or 'the current resource pool'}, the minimum allowed slice for {arch} is {min_allowed} chips, but you requested {num_cores}. To avoid an instant fragmentation rejection, request at least {arch}-{min_allowed}.")
                    if cores in TORUS_2D_MAP:
                        cores = TORUS_2D_MAP[cores]
                
                res_name = FISH_MAP.get(arch_lower, arch_lower)
            else:
                arch, cores = tpu_str.split('=', 1) if '=' in tpu_str else (tpu_str, "")
                res_name = FISH_MAP.get(arch.lower(), arch.lower())
            
            req_kwargs = {res_name: int(cores) if cores.isdigit() else cores}
            # /tmp on a Borg task is a RAM disk sized by this requirement, and
            # the default is far too small to stage a dataset into. Every task
            # of a multi-task TPU job copies its own private copy, so an
            # under-sized value shows up as `OSError: [Errno 28] No space left
            # on device` mid-download. 16 GiB matches what //third_party/py/maxtext
            # requests (xm_launch.py:238).
            req_kwargs['tmp_ram_fs'] = _TMP_RAM_FS_GIB.value * xm.GiB
            if _RAM_GIB.value > 0:
                req_kwargs['ram'] = int(_RAM_GIB.value * xm.GiB)
            if alloc_str:
                req_kwargs['allocator'] = alloc_str
            if _CELL.value:
                req_kwargs['location'] = _CELL.value
            tier_val = None
            for a in argv[1:]:
                if a.startswith('--tier='):
                    tier_val = a.split('=', 1)[1].upper()
                    if tier_val == 'PROD':
                        req_kwargs['service_tier'] = xm.ServiceTier.PROD
                    elif tier_val == 'BATCH':
                        req_kwargs['service_tier'] = xm.ServiceTier.BATCH
            job_requirements = xm.JobRequirements(**req_kwargs)
            
            # Borg's BorgScheduling defaults are max_task_failures=0 /
            # max_per_task_failures=0, i.e. "never restart". A preemption is
            # itself a free failure that does not count, but when a TPU gang is
            # torn apart the non-zero task exit IS counted as a FAILURE, and
            # CanStartTask() then declares the job dead (borg .../job.cc).
            # Result: one preemption kills the whole experiment.
            #
            # This applies to PROD too -- PROD is not immune to preemption:
            # xid 274552915 (PROD) died to SLICE_DEFRAGMENTATION, which is the
            # EQUAL-priority defrag path, not the higher-priority one.
            #
            # See //depot/google3/third_party/py/xmanager/xm_abc/executors.py
            # (class BorgScheduling) and go/borg-configure-schedule#task-failure-limits.
            scheduling = xm_abc.BorgScheduling(
                max_task_failures=_MAX_TASK_FAILURES.value,
                max_per_task_failures=_MAX_PER_TASK_FAILURES.value,
                task_failure_credit_period=_FAILURE_CREDIT_PERIOD.value,
            )

            if package_mode == "bazel":
                executor = xm_abc.Borg(
                    requirements=job_requirements,
                    scheduling=scheduling,
                    # Let colleagues (and future you) read this job's logs
                    # without an ACL dance.
                    logs_read_access_roles=['all'],
                )
            else:
                executor = xm_abc.Gcp(requirements=job_requirements)
            executors.append(executor)
            # Any TPU accelerator in the request means the job is multi-task and
            # needs the JAX coordination flags injected below.
            if res_name in FISH_MAP.values() or res_name.startswith('tpu'):
                is_tpu_job = True

        final_executor = xm.Fallback(executors) if len(executors) > 1 else executors[0]
    
        xid = experiment.experiment_id
        import time
        time_str = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"xid_{xid}_{time_str}_{exp_name}"
    
        import os
        import json
        import fcntl
        mapping_file = os.path.expanduser("~/.tpu_jobs.json")

        def read_mapping():
            if not os.path.exists(mapping_file):
                return {}
            try:
                with open(mapping_file, "r") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    data = json.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return data
            except Exception:
                return {}

        def update_mapping(xid, info):
            # MERGE (do not overwrite): xm_launcher runs before tpu_wrapper.sh's
            # own registration snippet, so we must preserve any pre-existing
            # tier/alloc/retry_count fields that another writer might set.
            try:
                with open(mapping_file, "a+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.seek(0)
                    content = f.read()
                    data = {}
                    if content:
                        try:
                            data = json.loads(content)
                        except ValueError:
                            data = {}
                    key = str(xid)
                    existing = data.get(key, {})
                    # Only fill in fields that are missing / empty in the existing entry,
                    # so a later writer with fresher tier/alloc info wins gracefully too.
                    merged = dict(existing)
                    for k, v in info.items():
                        # Overwrite empty / missing values; keep non-empty existing.
                        if merged.get(k) in (None, "", 0) or k not in merged:
                            merged[k] = v
                    data[key] = merged
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
                    fcntl.flock(f, fcntl.LOCK_UN)
            except Exception as e:
                print(f"Warning: could not write mapping file: {e}")

        if _RESUME_XID.value:
            # LOOK IN THE ARCHIVE TOO. `tpu clear` advertises itself as archiving
            # rather than deleting -- it moves entries to ~/.tpu_jobs_legacy.json --
            # but resume only consulted the live registry, so clearing a finished
            # run quietly made it un-resumable. The failure was not even a clear
            # message: it fell through to the long-dead ~/xm_job_to_bucket/ path
            # and raised FileNotFoundError on a file nothing has written since
            # 2026-07-26.
            want = str(_RESUME_XID.value)
            bucket_cp_path = ""
            for source in (read_mapping(), _read_legacy_mapping()):
                if want in source:
                    bucket_cp_path = source[want].get("bucket_cp_path", "")
                    if bucket_cp_path:
                        break
            if not bucket_cp_path:
                # Last resort: the pre-2026-07-26 one-file-per-xid layout.
                legacy_file = os.path.join(
                    os.path.expanduser("~/xm_job_to_bucket"), want)
                if not os.path.exists(legacy_file):
                    raise SystemExit(
                        f"--resume_xid={want}: no checkpoint bucket recorded for that "
                        f"experiment.\nLooked in ~/.tpu_jobs.json, "
                        f"~/.tpu_jobs_legacy.json and {legacy_file}.\n"
                        f"Pass --bucket=<its bucket_cp_path> explicitly if you know it.")
                with open(legacy_file, "r") as f:
                    bucket_cp_path = f.read().strip()
            vm_workdir = f"/tmp/eqr_log/resume_{xid}_{time_str}_{project_name}_{exp_name}"
        else:
            bucket_cp_path = f"{_local_bucket()}/logs/{project_name}/{folder_name}"
            vm_workdir = f"/tmp/eqr_log/{folder_name}"
            
        update_mapping(xid, {
            "bucket_cp_path": bucket_cp_path,
            "logdir": os.environ.get("TPU_LOGDIR", ""),
            "stagedir": os.environ.get("TPU_STAGEDIR", ""),
            "exp_name": exp_name,
            "tpu_type": _TPU_TYPE.value,
            # Recorded so `tpu check` can tell "preempted, will retry" from
            # "preempted, restart budget spent".
            "max_task_failures": _MAX_TASK_FAILURES.value,
        })
    
        config_path_arg = f"configs/load_config.py:{_CONFIG.value}"
        if package_mode == "bazel":
            config_path_arg = f"{pkg_path}/configs/load_config.py:{_CONFIG.value}"

        # NOTE: do NOT inject `--config.checkpoint_path`. `configs/default.py`
        # has no such field and `main.py` declares the config flag with
        # lock_config=True, so passing it makes every job die at startup.
        # The checkpoint location travels as the LOAD_FROM env var instead
        # (see below), matching unified_infra's convention
        # (infra/runjob.py:137-141) and the contract main.py already
        # implements in _ENV_CONFIG_OVERRIDES.
        executable_args = {
            'config': config_path_arg,
            'workdir': vm_workdir,
        }

        # Multi-host JAX coordination. Without these every task believes it is a
        # standalone task 0, and `jax.distributed.initialize()` blocks forever
        # waiting for peers that never announce themselves -- a hang, not an
        # error, so the job burns its whole deadline and dies with no useful
        # message. xm_jax fills them from Borg tokens at runtime:
        #   jax_controller_address -> get_job_bns_prefix() + "/0:jax"
        #   jax_num_tasks          -> replicas
        #   jax_task_id            -> %task%
        # See //depot/google3/third_party/py/xmanager/contrib/internal/xm_jax.py.
        if is_tpu_job:
            executable_args.update(xm_jax.JaxFlags().flags())
            # Prefer failing over running degraded: an ICI-resilient slice
            # costs ~35% throughput, and being rescheduled onto a healthy slice
            # beats finishing 1.5x slower. (mesh_diffusion launch_lib.py:372)
            executable_args['deepsea_ici_resilient'] = False
            # xm_jax's default controller address is a bare
            # `get_job_bns_prefix()`, which is only resolvable from inside the
            # job's own BCL scope. Qualify it with this job's name so the token
            # resolves regardless of how the experiment is structured -- the
            # same thing every production launcher does
            # (e.g. //learning/brain/experimental/jax_data/.../pst_trainer_launcher.py:272,
            # //third_party/py/scenic/google/xm/launch_xm.py:689).
            executable_args['jax_controller_address'] = xm_abc.RESTRICTED_BorgToken(
                f'{_JOB_NAME}.get_job_bns_prefix() + "/0:jax"'
            )
        
        for arg in argv[1:]:
            if arg.startswith('--'):
                if 'config.wandb' in arg or 'tpu_type' in arg or 'workdir' in arg or 'resume_xid' in arg:
                   continue
                # Resume/eval selectors travel as env vars, not config flags.
                # --tmp_ram_fs_gib is consumed HERE (it sizes the Borg RAM
                # disk in req_kwargs above); forwarding it on would hand the
                # application a flag it never declares. Harmless only if the
                # binary parses with known_only=True -- a job with a locked
                # config schema dies at startup instead.
                if arg.startswith(('--cell=', '--load_from=', '--config.load_from=',
                                   '--wandb_resume_id=', '--config.wandb_resume_id=',
                                   '--borg_max_task_failures=', '--borg_max_per_task_failures=',
                                   '--tmp_ram_fs_gib=', '--ram_gib=')):
                    continue
                key_val = arg[2:].split('=', 1)
                if len(key_val) == 2:
                    executable_args[key_val[0]] = key_val[1]
                else:
                    executable_args[key_val[0]] = ""

        # Resume/eval context goes through the environment, following
        # unified_infra (infra/runjob.py:137-141: "the training code reads
        # $LOAD_FROM / $WANDB_RESUME_ID; env vars are easier to adopt
        # group-wide than threading flags through every config"). main.py's
        # _ENV_CONFIG_OVERRIDES already consumes exactly these names, and env
        # wins over any seed value written in the yaml config.
        job_env_vars = {'PYTHONPATH': pkg_path}
        load_from = _LOAD_FROM.value
        # NOTE: --resume_xid deliberately does NOT set LOAD_FROM.
        #
        # It used to set it to f"{bucket_cp_path}/checkpoints", which is the
        # PARENT of the per-step directories. orbax restores a single
        # checkpoint dir, so it looked for `<...>/checkpoints/state` and died
        # with `FileNotFoundError: Checkpoint at .../checkpoints/state not
        # found.` (XID 275793223 attempt 2). Appending a step would be no
        # better: the launcher would have to guess which step survived.
        #
        # main.py::_apply_borg_autoresume already solves this correctly from
        # inside the job -- it enumerates $CHECKPOINT_BUCKET/checkpoints,
        # skips any directory without extra.json (written last, so its absence
        # marks a torn write), and resumes from the highest surviving step.
        # Crucially it SKIPS ITSELF when LOAD_FROM is set, treating that as an
        # explicit user request. So setting LOAD_FROM here did double damage:
        # it passed an unusable path AND disabled the mechanism that would have
        # picked the right one. Since --resume_xid reuses the same XID, and
        # CHECKPOINT_BUCKET is derived from the XID, the job lands on the same
        # prefix and rediscovery just works.
        if load_from:
            job_env_vars['LOAD_FROM'] = load_from
        if _WANDB_RESUME_ID.value:
            job_env_vars['WANDB_RESUME_ID'] = _WANDB_RESUME_ID.value
        # Where the job should persist its own checkpoints. workdir lives on
        # the task's local disk, which is wiped on every Borg task restart, so
        # a durable copy has to go to GCS for a restart to be able to resume.
        job_env_vars['CHECKPOINT_BUCKET'] = bucket_cp_path
        
        if package_mode == "bazel":
            (executable,) = experiment.package(
                [xm.bazel_binary(
                    label=target_label,
                    bazel_args=["--define=PYTYPE=FALSE", "--norun_validations"],
                    executor_spec=final_executor.Spec(),
                    args=executable_args,
                    env_vars=job_env_vars,
                )]
            )
        else: # python mode default
            base_image_accel = executors[0].requirements.accelerator
            (executable,) = experiment.package(
                [xm.python_container(
                    path='.',
                    base_image=framework_defaults.base_image('jax', base_image_accel),
                    entrypoint=xm.ModuleName('main'),
                    use_deep_module=True,
                    executor_spec=final_executor.Spec(),
                    args=executable_args,
                    # Same LOAD_FROM / WANDB_RESUME_ID contract as the bazel
                    # path, minus PYTHONPATH (use_deep_module handles imports).
                    env_vars={k: v for k, v in job_env_vars.items() if k != 'PYTHONPATH'},
                )]
            )
    
        # Args must be attached to the JOB, not only to the packageable.
        # `experiment.package(args=...)` records defaults on the executable,
        # but what Borg actually launches is built from the Job. Flags passed
        # only at package time can therefore go missing at runtime -- that is
        # how the xm_jax coordination flags were silently dropped, leaving
        # `jax.distributed.initialize()` to die with
        # "ValueError: coordinator_address should be defined."
        # Compare //depot/google3/third_party/py/maxtext/xm_launch.py, which
        # passes args to xm.Job.
        job = xm.Job(executable, final_executor, args=executable_args, name=_JOB_NAME)
        experiment.add(job)


if __name__ == '__main__':
    import sys
    # Same canonical map as in main(); duplicated here to avoid a forward ref.
    _GROUP_MAP = {
        '1': 'group:deepmind-dynamic/gdm-resources-prod-shared-users-dynamic',
        '2': 'group:deepmind-dynamic/gdm-viscam-goflow-dynamic',
        '3': 'group:deepmind-dynamic/gdm-viscam-interns-dynamic',
        '4': 'group:deepmind-dynamic/viscam-interns',
        '5': 'group:deepmind-dynamic/vqfree-xm',
        '6': 'group:dm/deepmind-large-scale-workshop',
        '7': 'group:dm/dm-resources-prod-shared',
        '8': 'group:gdm-aux/brain-vasp-shared-user-xm',
        '9': 'group:deepmind-dynamic/fr-dna-grand-challenge-team-resource',
    }
    new_argv = []
    for arg in sys.argv:
        if arg.startswith('--group='):
            group_val = arg.split('=', 1)[1]
            alloc = _GROUP_MAP.get(group_val, f'group:{group_val}')
            new_argv.append(f'--xm_resource_alloc={alloc}')
        else:
            new_argv.append(arg)
    sys.argv = new_argv
    app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
