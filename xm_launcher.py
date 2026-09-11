r"""Launcher for running EqR-jax on TPUs using XManager.

Usage example:
xmanager launch xm_launch.py -- \
  --xm_resource_alloc="group:gdm-aux/brain-vasp-shared-user-xm" \
  --tpu_type="v4-64,v5p-32" \
  --config="remote_run" \
  --workdir="~/logs/eqr-run"
"""
import os
import re
import subprocess
import sys

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

_PUBLISH_GCS = flags.DEFINE_bool(
    'publish_gcs', None,
    'Publish this run (code snapshot + checkpoints) to the GCS rendezvous '
    'bucket so the close-loop evaluator on the GPU VM can read it. UNSET '
    '(default) auto-detects: on when the config sets `dataset.task`, i.e. a '
    'RoboTwin run. Pass --publish_gcs / --nopublish_gcs to force either way. '
    'The primary checkpoint stream still goes to --bucket (CNS) regardless; '
    'this only adds a copy for a reader that cannot reach CNS.'
)

_BUCKET = flags.DEFINE_string(
    'bucket', '/cns/yutulpz-d/home/qiaos/eqr_data',
    'Durable root for checkpoints and mirrored logs. Defaults to CNS because a '
    'Borg job runs as <user>@prod.google.com, a different IAM principal from '
    'the <user>@google.com that owns our GCS bucket -- every gs:// write from a '
    'TPU worker fails with ACCESS_DENIED. Pass a gs:// path only if that bucket '
    'grants access to the prod identity.'
)
# WHERE A JOB'S CHECKPOINTS GO, resolved from the cell it lands in.
#
# Checkpoints are written from the TPU workers, so the bucket has to be near
# THEM, not near wherever the default happens to point. Getting this wrong is
# not a mild slowdown: XID 275990419 ran on `yuskedq` (metro ske, continent EU)
# while writing to `yutulpz` (metro tul, NA), and orbax reported 10 MiB/s, ~10s
# of BLOCKED TPU per save plus 33-56s of background flush. Its duty cycle fell
# to 0.082, below the 0.20 WIM pruning threshold, and the job was deleted.
# XID 284145906 died the same way on `yukulwh` (metro kul, ASIA).
#
# THE OLD TABLE MAPPED CELLS AND SILENTLY DEFAULTED. It listed 18 cells and any
# other cell fell through to `--bucket`'s default (`/cns/yutulpz-d`, metro tul,
# North America). That fall-through is the mechanism that killed the two jobs
# above: an unlisted cell is not a cell near tul, it is a cell we know nothing
# about, and 88% of the cells the scheduler can price were unlisted.
#
# THIS TABLE IS KEYED BY METRO, MEASURED, AND FAIL-CLOSED:
#   * cell -> metro comes from `mach_locality -k metro`, not from the cell's
#     name (the name-based guess was wrong for 26 of 57 cells);
#   * metro -> storage cell is the group's flex registration, verified with
#     `flex.par list_ceiling -s colossus -g deepmind-resources-colossus -l <c>`
#     ("Number of registrations found: 0" for an unregistered cell). NOT with
#     `fileutil quota`, which reports a plausible 500.00G for an unregistered
#     group because that is the default bucket it falls through to;
#   * a cell whose metro has no registration RAISES rather than defaulting.
# Collapsing 18 cell rows into 10 metro rows changes NO existing answer -- the
# old table was verified to be exactly this function of the measured metro.
#
# WHY THE ROWS ARE COPIED HERE instead of imported: this file is rsynced into a
# stagedir and built there, so it cannot import from ~/work/tpu_cmd. The copy is
# kept honest by `_assert_locality_matches_source()` below, which diffs it
# against the source of truth when that is reachable and FAILS on a mismatch --
# a silent drift between the two is exactly what this whole change removes.
#
# Regenerate with:
#   python3 ~/work/tpu_cmd/google3_tpu_utils/remeasure_cell_locality.py --write
#   python3 ~/work/tpu_cmd/google3_tpu_utils/sync_launcher_locality.py
_LOCALITY_MEASURED_AT = '2026-08-28T17:37:46Z'
_LOCALITY_SOURCE = os.path.expanduser(
    '~/work/tpu_cmd/google3_tpu_utils/cell_locality.py')

# cell -> (metro, continent). Compute cells only; storage cells live in
# _METRO_STORAGE_CELL below.
_CELL_LOCALITY = {
    'lcbomp':      ('bom', 'ap'),
    'ly':          ('bom', 'ap'),
    'wu':          ('icn', 'ap'),
    'yukulwh':     ('kul', 'ap'),
    'rx':          ('nrt', 'ap'),
    'sd':          ('sin', 'ap'),
    'se':          ('sin', 'ap'),
    'sf':          ('sin', 'ap'),
    'sg':          ('sin', 'ap'),
    'sh':          ('sin', 'ap'),
    'si':          ('sin', 'ap'),
    'sj':          ('sin', 'ap'),
    'sk':          ('sin', 'ap'),
    'sl':          ('sin', 'ap'),
    'sm':          ('sin', 'ap'),
    'sn':          ('sin', 'ap'),
    'so':          ('sin', 'ap'),
    'lcsydv':      ('syd', 'ap'),
    'ta':          ('tpe', 'ap'),
    'tb':          ('tpe', 'ap'),
    'tc':          ('tpe', 'ap'),
    'td':          ('tpe', 'ap'),
    'tg':          ('tpe', 'ap'),
    'th':          ('tpe', 'ap'),
    'tl':          ('tpe', 'ap'),
    'tm':          ('tpe', 'ap'),
    'tp':          ('tpe', 'ap'),
    'rc':          ('bll', 'eu'),
    'rd':          ('bll', 'eu'),
    'wb':          ('bru', 'eu'),
    'wd':          ('bru', 'eu'),
    'we':          ('bru', 'eu'),
    'wf':          ('bru', 'eu'),
    'wg':          ('bru', 'eu'),
    'wh':          ('bru', 'eu'),
    'wi':          ('bru', 'eu'),
    'wq':          ('bru', 'eu'),
    'yubrupd-c':   ('bru', 'eu'),
    'ra':          ('dhr', 'eu'),
    'rb':          ('dhr', 'eu'),
    'dg':          ('dub', 'eu'),
    'di':          ('dub', 'eu'),
    'dj':          ('dub', 'eu'),
    'lcfrai':      ('fra', 'eu'),
    'ea':          ('grq', 'eu'),
    'eb':          ('grq', 'eu'),
    'ec':          ('grq', 'eu'),
    'ed':          ('grq', 'eu'),
    'ef':          ('grq', 'eu'),
    'ei':          ('grq', 'eu'),
    'ej':          ('grq', 'eu'),
    'el':          ('grq', 'eu'),
    'en':          ('grq', 'eu'),
    'eq':          ('grq', 'eu'),
    'lclhrb':      ('lhr', 'eu'),
    'sv':          ('lhr', 'eu'),
    'yulhrp':      ('lhr', 'eu'),
    'yulhrs':      ('lhr', 'eu'),
    'la':          ('lpp', 'eu'),
    'lb':          ('lpp', 'eu'),
    'le':          ('lpp', 'eu'),
    'lg':          ('lpp', 'eu'),
    'lh':          ('lpp', 'eu'),
    'li':          ('lpp', 'eu'),
    'lj':          ('lpp', 'eu'),
    'lk':          ('lpp', 'eu'),
    'lo':          ('lpp', 'eu'),
    'lq':          ('lpp', 'eu'),
    'lt':          ('lpp', 'eu'),
    'lu':          ('lpp', 'eu'),
    'yulpptr':     ('lpp', 'eu'),
    'yuskedq':     ('ske', 'eu'),
    'ym':          ('atl', 'na'),
    'yo':          ('atl', 'na'),
    'yq':          ('atl', 'na'),
    'ys':          ('atl', 'na'),
    'lcausi':      ('aus', 'na'),
    'lcausr':      ('aus', 'na'),
    'ib':          ('cbf', 'na'),
    'if':          ('cbf', 'na'),
    'ig':          ('cbf', 'na'),
    'iq':          ('cbf', 'na'),
    'is':          ('cbf', 'na'),
    'it':          ('cbf', 'na'),
    'ix':          ('cbf', 'na'),
    'iy':          ('cbf', 'na'),
    'iz':          ('cbf', 'na'),
    'jb':          ('cbf', 'na'),
    'je':          ('cbf', 'na'),
    'jg':          ('cbf', 'na'),
    'ji':          ('cbf', 'na'),
    'jj':          ('cbf', 'na'),
    'jn':          ('cbf', 'na'),
    'jo':          ('cbf', 'na'),
    'jp':          ('cbf', 'na'),
    'jq':          ('cbf', 'na'),
    'js':          ('cbf', 'na'),
    'jt':          ('cbf', 'na'),
    'jz':          ('cbf', 'na'),
    'ny':          ('cbf', 'na'),
    'nz':          ('cbf', 'na'),
    'yucbfaa':     ('cbf', 'na'),
    'yucbfab':     ('cbf', 'na'),
    'yucbfac':     ('cbf', 'na'),
    'yucbfad':     ('cbf', 'na'),
    'yucbfad-c-staging': ('cbf', 'na'),
    'yucbfcd':     ('cbf', 'na'),
    'yucbfiv':     ('cbf', 'na'),
    'yucbflq':     ('cbf', 'na'),
    'yucbfpv':     ('cbf', 'na'),
    'yucbfrl':     ('cbf', 'na'),
    'yucbfsl':     ('cbf', 'na'),
    'yucbfsr':     ('cbf', 'na'),
    'yucbful':     ('cbf', 'na'),
    'yucbfwv':     ('cbf', 'na'),
    'ue':          ('chs', 'na'),
    'uj':          ('chs', 'na'),
    'ux':          ('chs', 'na'),
    'uy':          ('chs', 'na'),
    'vj':          ('chs', 'na'),
    'vk':          ('chs', 'na'),
    'vl':          ('chs', 'na'),
    'vz':          ('chs', 'na'),
    'yuchspe':     ('chs', 'na'),
    'yuchstz':     ('chs', 'na'),
    'ma':          ('ckv', 'na'),
    'mb':          ('ckv', 'na'),
    'md':          ('ckv', 'na'),
    'me':          ('ckv', 'na'),
    'mf':          ('ckv', 'na'),
    'mg':          ('ckv', 'na'),
    'mh':          ('ckv', 'na'),
    'mj':          ('ckv', 'na'),
    'yuckvax':     ('ckv', 'na'),
    'ga':          ('cmh', 'na'),
    'gb':          ('cmh', 'na'),
    'gh':          ('cmh', 'na'),
    'gl':          ('cmh', 'na'),
    'gm':          ('cmh', 'na'),
    'go':          ('cmh', 'na'),
    'rg':          ('cmh', 'na'),
    'yucmhaa':     ('cmh', 'na'),
    'yucmhab':     ('cmh', 'na'),
    'yucmhcg':     ('cmh', 'na'),
    'yucmhfq':     ('cmh', 'na'),
    'yucmhgs':     ('cmh', 'na'),
    'yucmhnb':     ('cmh', 'na'),
    'yucmhps':     ('cmh', 'na'),
    'yucmhqa':     ('cmh', 'na'),
    'yucmhsu':     ('cmh', 'na'),
    'yucmhty':     ('cmh', 'na'),
    'yucmhwf':     ('cmh', 'na'),
    'lcdfwlf':     ('dfw', 'na'),
    'rq':          ('dfw', 'na'),
    'rr':          ('dfw', 'na'),
    'rs':          ('dfw', 'na'),
    'rt':          ('dfw', 'na'),
    'rw':          ('dfw', 'na'),
    'yudfwra':     ('dfw', 'na'),
    'yudfwra-c':   ('dfw', 'na'),
    'pw':          ('dls', 'na'),
    'px':          ('dls', 'na'),
    'py':          ('dls', 'na'),
    'pz':          ('dls', 'na'),
    'ts':          ('dls', 'na'),
    'tt':          ('dls', 'na'),
    'yufwahd':     ('fwa', 'na'),
    'yufwakf':     ('fwa', 'na'),
    'bh':          ('iad', 'na'),
    'bi':          ('iad', 'na'),
    'bk':          ('iad', 'na'),
    'pd':          ('iad', 'na'),
    'wv':          ('iad', 'na'),
    'ww':          ('iad', 'na'),
    'yuiadrs':     ('iad', 'na'),
    'yuiadtq':     ('iad', 'na'),
    'dd':          ('las', 'na'),
    'dl':          ('las', 'na'),
    'dy':          ('las', 'na'),
    'dz':          ('las', 'na'),
    'qc':          ('mrn', 'na'),
    'qn':          ('mrn', 'na'),
    'qo':          ('mrn', 'na'),
    'qr':          ('mrn', 'na'),
    'yumrnel':     ('mrn', 'na'),
    'yuphxej':     ('phx', 'na'),
    'yuphxer':     ('phx', 'na'),
    'yuphxrp':     ('phx', 'na'),
    'ro':          ('rno', 'na'),
    'yurnoaa':     ('rno', 'na'),
    'yurnolb':     ('rno', 'na'),
    'yurnoyc':     ('rno', 'na'),
    'na':          ('tul', 'na'),
    'nf':          ('tul', 'na'),
    'nk':          ('tul', 'na'),
    'nl':          ('tul', 'na'),
    'nm':          ('tul', 'na'),
    'nn':          ('tul', 'na'),
    'oa':          ('tul', 'na'),
    'od':          ('tul', 'na'),
    'oe':          ('tul', 'na'),
    'oi':          ('tul', 'na'),
    'oj':          ('tul', 'na'),
    'ok':          ('tul', 'na'),
    'oq':          ('tul', 'na'),
    'ot':          ('tul', 'na'),
    'ow':          ('tul', 'na'),
    'oz':          ('tul', 'na'),
    'pa':          ('tul', 'na'),
    'pb':          ('tul', 'na'),
    'yutulis':     ('tul', 'na'),
    'yutulpz':     ('tul', 'na'),
    'yutulrf':     ('tul', 'na'),
    'gc':          ('uos', 'na'),
    'gd':          ('uos', 'na'),
    'gd-c':        ('uos', 'na'),
    'ge':          ('uos', 'na'),
    'gg':          ('uos', 'na'),
    'lcyulk':      ('yul', 'na'),
    'ce':          ('scl', 'sa'),
    'cf':          ('scl', 'sa'),
    'cg':          ('scl', 'sa'),
    'cj':          ('scl', 'sa'),
    'lcscld':      ('scl', 'sa'),
}

# metro -> the CNS cell the GROUP is registered in. One row per metro
# because every cell in a metro shares its storage.
_METRO_STORAGE_CELL = {
    'cbf':   'is-d',
    'ckv':   'mb-d',
    'cmh':   'go-d',
    'dfw':   'rs-d',
    'grq':   'el-d',
    'las':   'dl-d',
    'lpp':   'li-d',
    'mrn':   'qo-d',
    'sin':   'si-d',
    'tul':   'oi-d',
}

# Metros with NO group registration: a write here lands on the PERSONAL
# 500 GiB per-cell ceiling. Named rather than omitted so the error can
# say WHICH kind of "no" it is.
_PERSONAL_ONLY_METROS = {
    'phx':   'yuphxrp-d',
    'ske':   'yuskedq-d',
}


def _metro_of(cell):
    """Measured metro for `cell`, or None. Never guesses from the name."""
    row = _CELL_LOCALITY.get((cell or '').strip().lower())
    return row[0] if row else None


def _continent_of(cell):
    """Measured continent for `cell`, or None."""
    row = _CELL_LOCALITY.get((cell or '').strip().lower())
    return row[1] if row else None


def _storage_cell_of(cell):
    """The CNS cell co-located with `cell`, or None if none is registered."""
    metro = _metro_of(cell)
    if metro is None:
        return None
    return _METRO_STORAGE_CELL.get(metro) or _PERSONAL_ONLY_METROS.get(metro)


def _assert_locality_matches_source():
    """Fail if this embedded copy has drifted from the shared snapshot.

    Runs only where the source file is reachable (an interactive launch from the
    checkout); inside a stagedir build it is absent and the check is skipped --
    which is safe because the stagedir copy was rsynced from a checkout where
    this same assertion had already passed at launch time.

    The check is a DIFF, not a version stamp: a stamp can be bumped without the
    rows changing and rows can change without the stamp moving.
    """
    try:
        with open(_LOCALITY_SOURCE) as handle:
            source = handle.read()
    except OSError:
        return  # not reachable from here; nothing to compare against
    namespace = {}
    try:
        exec(compile(source, _LOCALITY_SOURCE, 'exec'), namespace)  # noqa: S102
        truth = namespace['_MEASURED']
        truth_storage = namespace['_METRO_STORAGE_CELL']
        truth_personal = namespace['_PERSONAL_ONLY_METROS']
    except Exception as exc:  # noqa: BLE001 - a broken source must not be silent
        raise SystemExit(
            '[locality] cannot read the cell-locality source of truth '
            f'{_LOCALITY_SOURCE}: {exc}. Refusing to launch with an '
            'unverifiable bucket map -- fix the file, or delete it to fall '
            'back to this launcher\'s embedded copy.')
    want = {c: (r[0], r[1]) for c, r in truth.items() if not c.endswith('-d')}
    problems = []
    for cell in sorted(set(want) | set(_CELL_LOCALITY)):
        mine, theirs = _CELL_LOCALITY.get(cell), want.get(cell)
        if mine != theirs:
            problems.append(f'  {cell}: launcher={mine} source={theirs}')
    if truth_storage != _METRO_STORAGE_CELL:
        problems.append(f'  metro->storage: launcher={_METRO_STORAGE_CELL} '
                        f'source={truth_storage}')
    if truth_personal != _PERSONAL_ONLY_METROS:
        problems.append(f'  personal-only: launcher={_PERSONAL_ONLY_METROS} '
                        f'source={truth_personal}')
    if problems:
        raise SystemExit(
            '[locality] this launcher\'s embedded cell map has DRIFTED from '
            f'{_LOCALITY_SOURCE}:\n' + '\n'.join(problems[:20])
            + (f'\n  ... and {len(problems) - 20} more' if len(problems) > 20 else '')
            + '\n  Re-sync with: python3 '
              '~/work/tpu_cmd/google3_tpu_utils/sync_launcher_locality.py\n'
              '  Refusing to launch: a drifted bucket map is how a job ends up '
              'checkpointing to another continent.')

# WHERE A v7 JOB SHOULD PREFER TO LAND, best tier first.
#
# The scheduler ranks on capacity and price, and knows nothing about where our
# storage is -- which is how jobs kept landing in `ske`: most chips in the
# fleet, and the only metro of the four with NO team storage at all, so every
# byte lands on the personal 500 GiB ceiling. Exhausting that ceiling poisons
# every write in the cell, including the log mirror, which is how one run
# trained 130k steps and produced four 0-byte logs.
#
# So we express a PREFERENCE, not a ban. `ske` stays usable -- it is real
# capacity and the personal quota is fine as long as `tpu gc` keeps it swept --
# it just goes last.
#
# Tier 1: the three metros datasets are actually mirrored into
#         (research/v7_storage_placement.md). Compute, data and checkpoints all
#         local, charged to the group's PiB.
# Tier 2: other metros with PiB-scale group quota. Safe for checkpoints, but the
#         dataset may need staging first.
# Tier 3: no team storage in the metro; personal ceiling only. Last resort.
_CELL_TIERS = (
    ('cbf/tul/lpp (mirrored data, group quota)',
     ('yucbfiv', 'yucbful', 'yucbfwv', 'je', 'yutulpz', 'yulpptr')),
    ('other metros with group quota',
     ('sk', 'sn', 'so', 'yumrnel', 'el', 'mb', 'yudfwra', 'dl')),
    ('no team storage -- personal 500 GiB only',
     ('yuskedq',)),
)


def _preferred_cells(spec: str) -> list[str]:
    """Cells to allow, best-first. `spec` is 'auto', 'off', or a cell list."""
    spec = (spec or '').strip()
    if not spec or spec.lower() == 'off':
        return []
    if spec.lower() != 'auto':
        return [c.strip() for c in spec.split(',') if c.strip()]
    cells: list[str] = []
    for _, tier in _CELL_TIERS:
        cells.extend(tier)
    return cells


def _describe_cell_choice(cells: list[str]) -> None:
    """Print which tier each allowed cell came from, so the choice is auditable."""
    for label, tier in _CELL_TIERS:
        present = [c for c in cells if c in tier]
        if present:
            print(f"[locality]   {label}: {' '.join(present)}")


def _read_legacy_mapping():
    """Archived job registry (`tpu clear` moves entries here). {} if absent."""
    import json
    try:
        with open(os.path.expanduser("~/.tpu_jobs_legacy.json"), "r") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001 - the archive is optional
        return {}


# The path under a CNS cell root that this project's checkpoints live in. The
# shared locality layer owns WHICH CELL; the directory below it stays here.
_BUCKET_SUFFIX = 'home/qiaos/eqr_data'


def _local_bucket() -> str:
    """The durable root nearest the cell this job will run in, or exit.

    FAIL CLOSED, ON AN UNCONDITIONAL PATH. Every return below is either an
    explicit user choice or a bucket PROVEN co-located with a measured cell;
    there is no branch that falls through to a default prefix. The previous
    version returned `--bucket`'s default for any cell it did not recognise,
    which is how XID 284145906 (cell `yukulwh`, metro kul, ASIA) came to write
    its checkpoints to `/cns/yutulpz-d` in North America, stall on every save,
    and be deleted by the WIM pruner. Nothing in its logs said so, because the
    wrong path is a perfectly valid path.

    The guard is deliberately NOT nested inside an `if binary exists` or
    `if flag set` block: a protection reachable only on some paths is a
    protection that disappears exactly when something else has already gone
    wrong.
    """
    # 1. An explicitly passed --bucket is authoritative. The user named a
    #    location; it is not this function's business to second-guess it.
    if _BUCKET.present:
        return _BUCKET.value

    # 2. This launcher's embedded cell map must agree with the shared snapshot.
    #    Checked here, on the one path every unpinned launch takes.
    _assert_locality_matches_source()

    cell = (_CELL.value or '').strip()

    # 3. NO PINNED CELL. Under --cell_prefer the scheduler picks from a whole
    #    tier, so the landing cell is not knowable here and no bucket can be
    #    proven co-located with it. Refuse rather than ship a plausible guess.
    if not cell:
        raise SystemExit(
            '[locality] REFUSING to launch: no --cell is pinned, so the landing '
            'cell -- and therefore the metro its checkpoints must be written in '
            '-- is not knowable at submit time. Falling back to the default '
            f'bucket ({_BUCKET.value!r}) is what got XID 284145906 deleted: it '
            'wrote from Asia to North America, stalled the accelerator on every '
            'save, and the pruner removed it mid-run.\n'
            '  Fix, in order of preference:\n'
            '    * pass --cell=<cell> (tpu queue pins one for you by default), or\n'
            '    * pass --bucket=<root> if you have chosen a location on purpose.')

    # 4. A pinned cell resolves through the MEASURED map, or refuses.
    storage_cell = _storage_cell_of(cell)
    if storage_cell is None:
        metro = _metro_of(cell)
        if metro is None:
            why = (f'cell {cell!r} is not in the measured locality snapshot '
                   f'({len(_CELL_LOCALITY)} cells, measured '
                   f'{_LOCALITY_MEASURED_AT}), so no bucket can be proven '
                   f'co-located with it')
            how = ('re-measure with `python3 '
                   '~/work/tpu_cmd/google3_tpu_utils/remeasure_cell_locality.py '
                   '--write` (the cell may be newly turned up), then re-sync '
                   'this launcher with sync_launcher_locality.py')
        else:
            why = (f'cell {cell!r} is in metro {metro!r} (continent '
                   f'{_continent_of(cell)!r}), which has NO storage cell '
                   f'registered for the group')
            how = (f'register the group in a {metro!r} CNS cell and add it to '
                   f'_METRO_STORAGE_CELL, or run in a metro that has storage '
                   f'({", ".join(sorted(_METRO_STORAGE_CELL))})')
        raise SystemExit(
            f'[locality] REFUSING to launch: {why}. Refusing to fall back to '
            f'{_BUCKET.value!r} -- an out-of-metro checkpoint stream is not a '
            f'slowdown, it is a deletion (XID 275990419, XID 284145906).\n'
            f'  Fix: {how}; or pass --bucket=<root> explicitly.')

    bucket = f'/cns/{storage_cell}/{_BUCKET_SUFFIX}'
    metro = _metro_of(cell)
    # 5. PERSONAL-ONLY METROS ARE REFUSED, NOT WARNED ABOUT.
    #    This used to print a NOTE saying "fine for a smoke test" and launch
    #    anyway. That advice expired: the personal quota is at 468G of its
    #    500 GiB ceiling and its handle is poisoned, so a write there fails
    #    with resource_exhausted AND LEAVES A 0-BYTE FILE -- the loss looks
    #    like a file that exists, and `tpu check` still reports SUBMITTED.
    #    ★This is the worst of the three metro outcomes: a metro in neither
    #    dict SystemExits into an inert zero-work-unit shell, which is at
    #    least visibly broken; phx/ske launch, bill, and destroy their own
    #    output while looking healthy. A warning is the wrong instrument for
    #    a failure that already looks like success -- nobody reads a NOTE on
    #    a launch that appears to work.
    #    Escape hatch: an explicit --bucket returns at step 1 above and never
    #    reaches here, so a caller who has deliberately chosen a group-billed
    #    path in these metros is unaffected.
    if metro in _PERSONAL_ONLY_METROS:
        raise SystemExit(
            f'[locality] REFUSING to launch: cell {cell!r} is in metro '
            f'{metro!r}, which has NO GROUP storage registration -- writes '
            f'land on the PERSONAL {_PERSONAL_ONLY_METROS[metro]} quota '
            f'(500 GiB per cell, currently ~468G used and poisoned). A write '
            f'there fails with resource_exhausted and still leaves a 0-byte '
            f'file, so the job looks like it produced output.\n'
            f'  Fix, in order of preference:\n'
            f'    * run in a metro with group storage: '
            f'{", ".join(sorted(_METRO_STORAGE_CELL))}; or\n'
            f'    * pass --bucket=<group-billed root> explicitly if you have '
            f'chosen this location on purpose.')
    print(f'[locality] cell={cell} (metro {metro}, continent '
          f'{_continent_of(cell)}): co-located bucket {bucket}')
    return bucket


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
# These are REAL-FAILURE counters; Borg tracks preemptions/evictions against the
# separate max_task_evictions budget below.
_MAX_TASK_FAILURES = flags.DEFINE_integer(
    'borg_max_task_failures', -1,
    'Total non-eviction task failures tolerated across all tasks before the '
    'job is aborted. -1 means unlimited. This does not limit preemption '
    'restarts; use --borg_max_task_evictions for those.'
)
_MAX_PER_TASK_FAILURES = flags.DEFINE_integer(
    'borg_max_per_task_failures', 3,
    'Failures tolerated per individual task before that task is declared dead. '
    'Combined with the credit period below this reads as "recover from at most '
    'THREE failures per task every N seconds". Raised 1 -> 3 by operator order '
    '(2026-08-31 01:27Z). At 1, a single transient CUDA/NCCL hiccup on one rank '
    'killed a multi-hour multi-host run outright; the other two budgets '
    '(borg_max_task_failures / borg_max_task_evictions) are already -1 = '
    'unlimited, so this was the only counter that could end a healthy job. The '
    'credit period still decays the count, so a genuinely broken task -- one '
    'failing faster than the decay -- is still declared dead rather than '
    'restarted forever.'
)
_MAX_TASK_EVICTIONS = flags.DEFINE_integer(
    'borg_max_task_evictions', -1,
    'Total task evictions tolerated before the job is aborted. -1 means '
    'unlimited (the Borg default); 0 prevents an evicted/preempted task from '
    'being restarted. This is distinct from task failures.'
)
_FAILURE_CREDIT_PERIOD = flags.DEFINE_integer(
    'borg_failure_credit_period', 7200,
    'Every N seconds Borg decrements each live task\'s failure count, so a run '
    'is not killed by slow attrition of unrelated one-off failures.'
)


def _borg_overrides_for_eviction_budget(max_task_evictions: int):
    """Builds the narrow Borg override needed for an eviction budget.

    XManager's BorgScheduling exposes failure budgets, but not Borg's separate
    max_task_evictions field. Keep the default path shape-identical by emitting
    no override for -1 (Borg's unlimited default), and set only the missing
    scheduling field when the operator explicitly supplies a finite budget.
    """
    if max_task_evictions < -1:
        raise ValueError(
            '--borg_max_task_evictions must be -1 (unlimited) or non-negative; '
            f'got {max_task_evictions}.')
    if max_task_evictions == -1:
        return None
    borg_overrides = xm_abc.RESTRICTED_BorgOverrides()
    borg_overrides.scheduling.max_task_evictions = max_task_evictions
    return borg_overrides


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
# FLOAT, not int, and the reason is a scheduling one rather than a tidiness one.
# This RAM disk is real RAM and it is counted in the job's memory request. For a
# batch job the size of the request IS the queue wait: a task that stages one
# ~18 MB shard was asking for a whole GiB, which at 4 tasks was 74% of a 5.4 GiB
# ask and put the job 824-deep in a best-effort queue. An integer flag has 1 GiB
# as its floor, so there was no way to ask for the ~64 MiB actually needed.
_TMP_RAM_FS_GIB = flags.DEFINE_float(
    'tmp_ram_fs_gib', 0,
    'Size of the per-task RAM disk backing /tmp, in GiB. 0 (default) MEASURES '
    'the dataset the config names and sizes the disk to fit it; any positive '
    'value overrides that. Must exceed whatever the job stages locally '
    '(dataset copies, scratch). Fractional values are allowed and matter: this '
    'is RAM, it is charged to the job\'s memory request, and on a best-effort '
    'tier an oversized request is queue time.'
)

# What `tmp_ram_fs_gib: 0` resolves to when the dataset cannot be measured --
# an unreadable path, a config the launcher cannot parse, a corpus generated on
# the fly. The historical default, so a job that measures nothing behaves
# exactly as it did before auto-sizing existed.
_TMP_RAM_FS_FALLBACK_GIB = 16.0
# Multiplied onto the measured payload. The job writes the bytes it read plus
# scratch, and a RAM disk that is exactly the payload fails on the last chunk.
_TMP_RAM_FS_HEADROOM = 1.35
# Never ask for less than this even for a tiny corpus: the runfiles tree, logs
# and orbax scratch all live in the same /tmp.
_TMP_RAM_FS_FLOOR_GIB = 4.0
# Refuse to auto-size beyond this; past it something is wrong with the estimate
# and a silent 500 GiB memory request would never schedule. An explicit
# --tmp_ram_fs_gib still goes through, because the operator has looked.
_TMP_RAM_FS_CEILING_GIB = 96.0


def _measure_remote_dataset_gib(path: str) -> float:
    """GiB the job will stage from `path`, or 0.0 if it cannot be measured.

    RETURNS 0.0 RATHER THAN RAISING, and every caller treats 0.0 as "unknown"
    and falls back. This runs on the submit path of every job: a sizing helper
    that can abort a launch has negative value, exactly like the in-job metric
    guards (`wiki_agents/engineering.md`: do not let a diagnostic kill the thing
    it watches).

    Only `/cns/` is measured. A local path is not staged at all (the reader
    mmaps it in place), and `gs://` needs a different CLI that the submit path
    should not be shelling out to.

    `ls -l` PER SPLIT, NOT `du -s` ON THE ROOT. These corpora keep their
    generation shards beside the merged payload -- 90,002 directories holding
    450,010 files for one corpus -- and `du` walks all of it: measured, it had
    not returned after 90s and was killed, which would stall every launch by
    that much. The merged `.npy` files sit flat in each split, so a non-recursive
    listing that sums the file column is both complete and ~4s. Directories are
    skipped by the leading-`d` test, which is exactly what excludes `shards/`.
    """
    if not path or not path.startswith('/cns/'):
        return 0.0
    total = 0
    seen_any = False
    for split in ('train', 'test'):
        try:
            out = subprocess.run(
                ['fileutil', 'ls', '-l', f'{path}/{split}'],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:  # noqa: BLE001 -- never fail a launch on a size estimate
            continue
        if out.returncode != 0:
            continue
        for line in out.stdout.splitlines():
            if not line or line.startswith('d'):
                continue  # a directory -- `shards/`, which is not staged
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                total += int(parts[4])
                seen_any = True
            except ValueError:
                continue
    return total / float(1 << 30) if seen_any else 0.0


def _dataset_name_from_yaml() -> str:
    """`dataset.name` read TEXTUALLY out of the run's yaml, or ''.

    DELIBERATELY NOT `from configs import load_config`. This launcher is executed
    by `xmanager launch`, which runs it under a HERMETIC interpreter that has
    none of the project's dependencies -- so importing the project's config
    loader raises, the caller's `except` swallows it, and auto-sizing silently
    falls back to the default. That is exactly how the first version of this
    feature failed: it worked in every hand-run test (a conda python, in the
    stagedir, where the import succeeds) and did nothing at all under the real
    launcher, which is the only place it matters.

    A five-line scan of the yaml has no dependencies and cannot fail that way.
    It only has to find one key, and if the file is templated or absent the
    caller falls back exactly as before.
    """
    for candidate in (f'configs/{_CONFIG.value}_config.yml',
                      f'configs/{_CONFIG.value}_config.yaml'):
        try:
            with open(candidate, 'r') as handle:
                in_dataset = False
                for raw in handle:
                    line = raw.rstrip('\n')
                    if not line.strip() or line.lstrip().startswith('#'):
                        continue
                    if not line[:1].isspace():           # a top-level key
                        in_dataset = line.startswith('dataset:')
                        continue
                    if in_dataset and line.strip().startswith('name:'):
                        return line.split(':', 1)[1].strip().strip('\'"')
        except OSError:
            continue
    return ''


# Dataset alias -> CNS root, for sizing ONLY. A DUPLICATE of the project's own
# `dataset/data_util.py::DATASET_PATHS`, and duplicated on purpose: see
# `_dataset_name_from_yaml` for why this file cannot import that module.
#
# Drift here is SAFE BY CONSTRUCTION -- an alias this map does not know simply
# measures nothing and falls back to the default, which is what every job did
# before auto-sizing existed. It can never point a job at the wrong data: only
# the RAM disk size is derived from it.
#
# UNLISTED IS NOT HARMLESS WHEN THE CORPUS IS HUGE. The fallback is 16 GiB, so
# an alias missing here hands a 51 GiB corpus a 16 GiB RAM disk and the job dies
# mid-copy in `data_util._copy_file` with `[Errno 28] No space left on device`
# -- measured, xid 278952152. `[ramdisk] dataset not measurable` in the launch
# log IS that failure, printed 20 minutes before it happens: read it.
_SIZING_DATASET_ROOTS = {
    'Maze-period-easy': '/cns/is-d/home/qiaos/eqr_maze_settingA/maze-period-easy',
    'Maze-period-hard': '/cns/is-d/home/qiaos/eqr_maze_settingA/maze-period-hard',
    # Setting B-v3 open-loop. Measured staged sizes (payload only, `seeds.npy`
    # and `provenance.json` skipped by data_util._UNUSED_BY_TRAINING):
    # easy 50.9 GiB / mid 58.3 GiB / adv 68.8 GiB -- so adv x1.35 lands at
    # 92.8 GiB, just under the 96 GiB auto-size ceiling.
    'Maze-b3-easy': '/cns/is-d/home/qiaos/eqr_maze_settingB_v3/maze-b3-easy',
    'Maze-b3-mid': '/cns/is-d/home/qiaos/eqr_maze_settingB_v3/maze-b3-mid',
    'Maze-b3-adv': '/cns/is-d/home/qiaos/eqr_maze_settingB_v3/maze-b3-adv',
}


def _dataset_path_from_project() -> str:
    """The CNS root of the configured dataset, or '' when it cannot be resolved."""
    name = _dataset_name_from_yaml()
    if not name:
        return ''
    if name.startswith('/cns/'):     # a literal path in the yaml
        return name
    return _SIZING_DATASET_ROOTS.get(name, '')


# --------------------------------------------------------------------------
# RoboTwin close-loop eval: publish the run to a GCS rendezvous bucket.
#
# WHY A BUCKET AT ALL. The close-loop evaluator runs on a GCP A100 VM in
# project `viscam-cloud`. That VM cannot read CNS -- no LOAS credential and no
# `fileutil` (MEASURED on the box) -- and Borg cannot call the VM either: a
# Borg task cannot even CREATE an IPv4 socket (`AF_INET` -> OSError errno 97,
# MEASURED from inside a real task, XID 279103266), while the VM's public IP is
# IPv4-only. The one channel proven from both ends is GCS: 0.3 ms from Borg
# over IPv6, 120.8 MB/s to the VM. CNS stays the primary store; this is one
# extra copy for the single reader that cannot reach it.
#
# THE MAPPING RULE, in one place (`utils/gcs_publish.py` repeats it for the
# job side, and its tests pin it):
#
#     gs://<bucket>/runs/<xid>/
#         stagedir/      the immutable CitC snapshot this job was built from,
#                        uploaded ONCE here at launch
#         checkpoints/   step_<n>/... mirrored by the job after each save,
#                        `extra.json` LAST so a reader never sees a
#                        complete-looking checkpoint with weights in flight
#         meta.json      xid, exp_name, task, cns path, timestamps
#
# `stagedir/` and `checkpoints/` are SIBLINGS on purpose: the snapshot is
# immutable and the checkpoint stream is append-only, and nesting them makes
# "has the code changed?" unanswerable from a listing.
# --------------------------------------------------------------------------

#: us-east4 == the GPU VM's zone, so the VM reads without cross-region egress.
_ROBOTWIN_EVAL_BUCKET = 'gs://qiaos-robotwin-eval-us-east4'


def _robotwin_task_from_yaml() -> str:
    """`dataset.task` read TEXTUALLY out of the run's yaml, or ''.

    Same constraint as `_dataset_name_from_yaml`, and the same reason: this
    launcher runs under `xmanager launch`'s HERMETIC interpreter, which has
    none of the project's dependencies, so `from configs import load_config`
    raises and any auto-detection built on it silently does nothing. A textual
    scan cannot fail that way.

    A non-empty `dataset.task` IS the RoboTwin marker: `configs/dp_default.py`
    documents it as "the RoboTwin task to train on (e.g. click_bell)", and no
    maze or sudoku config sets it.
    """
    for candidate in (f'configs/{_CONFIG.value}_config.yml',
                      f'configs/{_CONFIG.value}_config.yaml'):
        try:
            with open(candidate, 'r') as handle:
                in_dataset = False
                for raw in handle:
                    line = raw.rstrip('\n')
                    if not line.strip() or line.lstrip().startswith('#'):
                        continue
                    if not line[:1].isspace():           # a top-level key
                        in_dataset = line.startswith('dataset:')
                        continue
                    if in_dataset and line.strip().startswith('task:'):
                        return line.split(':', 1)[1].strip().strip('\'"')
        except OSError:
            continue
    return ''


def _should_publish_to_gcs() -> bool:
    """True when this run should be published for close-loop eval.

    `--publish_gcs` is tri-state on purpose: unset means "decide from the
    config" (RoboTwin runs publish, nothing else does), and an explicit
    true/false overrides that -- so a one-off can publish without editing a
    config, and a RoboTwin run can opt OUT without the launcher arguing.
    """
    if _PUBLISH_GCS.value is not None:
        return bool(_PUBLISH_GCS.value)
    return bool(_robotwin_task_from_yaml())


def _publish_stagedir(xid: str, meta: dict) -> None:
    """Upload the CitC snapshot + meta.json once, at launch. Never raises.

    Runs on the WORKSTATION (which can read CitC and reach GCS), not in the
    job -- the job's container has neither the snapshot nor a reason to upload
    it. Failure is reported and ignored: a publish problem must not abort a
    launch that is otherwise fine, and the evaluator simply finds no run.
    """
    import subprocess
    stagedir = os.environ.get('TPU_STAGEDIR', '')
    root = f'{_ROBOTWIN_EVAL_BUCKET}/runs/{xid}'
    if not stagedir or not os.path.isdir(stagedir):
        print(f'[gcs-publish] no readable TPU_STAGEDIR ({stagedir!r}); '
              f'skipping the code snapshot upload for {xid}')
    else:
        print(f'[gcs-publish] {stagedir} -> {root}/stagedir')
        try:
            # `rsync -r` not `cp -r`: a relaunch onto the same XID re-uploads
            # only what changed, and the snapshot is typically unchanged.
            p = subprocess.run(
                ['gcloud', 'storage', '-q', 'rsync', '-r',
                 '-x', r'(^|/)(\.git|bazel-.*|__pycache__|wandb|logs)(/|$)',
                 stagedir, f'{root}/stagedir'],
                capture_output=True, text=True, timeout=900)
            if p.returncode != 0:
                print(f'[gcs-publish] WARNING: stagedir upload rc={p.returncode}: '
                      f'{(p.stderr or "").strip()[-400:]}')
            else:
                print('[gcs-publish] stagedir uploaded')
        except Exception as exc:  # noqa: BLE001 -- a publish must not fail a launch
            print(f'[gcs-publish] WARNING: stagedir upload failed: '
                  f'{type(exc).__name__}: {exc}')
    try:
        import json as _json
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
            _json.dump(meta, handle, indent=2, sort_keys=True, default=str)
            tmp = handle.name
        subprocess.run(['gcloud', 'storage', '-q', 'cp', tmp, f'{root}/meta.json'],
                       capture_output=True, text=True, timeout=120)
        os.unlink(tmp)
        print(f'[gcs-publish] meta.json -> {root}/meta.json')
    except Exception as exc:  # noqa: BLE001
        print(f'[gcs-publish] WARNING: meta.json upload failed: '
              f'{type(exc).__name__}: {exc}')


def _resolve_tmp_ram_fs_gib(config_obj) -> float:
    """The RAM disk this job needs, in GiB.

    WHY THIS IS MEASURED RATHER THAN RAISED FOR EVERYONE. The disk is real RAM
    charged to the job's memory request, and the request is the queue wait --
    one measured case put a job 824-deep in a best-effort queue for asking a
    whole GiB to stage an 18 MB shard. So "just make the default big" trades a
    rare crash for a permanent scheduling tax on every job. Measuring costs one
    `fileutil du` at submit time and charges each job for what it actually
    stages.

    The failure this replaces: staging a 36 GB corpus into the old fixed 16 GiB
    default dies with `OSError: [Errno 28] No space left on device` inside
    `data_util._copy_file`, mid-download, after the slice is already allocated.
    Borg retries it, so it burns the restart budget and reads as a hardware
    fault -- the surface errors were `Fish chip setup` and `Job terminated in
    state FAILURE`, neither of which names the disk.
    """
    explicit = _TMP_RAM_FS_GIB.value
    if explicit and explicit > 0:
        return float(explicit)

    dataset_path = _dataset_path_from_project()

    measured = _measure_remote_dataset_gib(dataset_path)
    if measured <= 0.0:
        print(f"[ramdisk] dataset not measurable ({dataset_path or 'unknown'}); "
              f"using {_TMP_RAM_FS_FALLBACK_GIB:g} GiB")
        return _TMP_RAM_FS_FALLBACK_GIB

    want = max(measured * _TMP_RAM_FS_HEADROOM, _TMP_RAM_FS_FLOOR_GIB)
    if want > _TMP_RAM_FS_CEILING_GIB:
        print(f"[ramdisk] measured {measured:.1f} GiB at {dataset_path}, which "
              f"wants {want:.1f} GiB -- above the {_TMP_RAM_FS_CEILING_GIB:g} GiB "
              f"auto-size ceiling. Capping; pass --tmp_ram_fs_gib to override.")
        want = _TMP_RAM_FS_CEILING_GIB
    print(f"[ramdisk] {dataset_path} measures {measured:.1f} GiB -> "
          f"/tmp sized {want:.1f} GiB (x{_TMP_RAM_FS_HEADROOM} headroom)")
    return want
# Every job this launcher has ever submitted ran as exactly ONE task, because
# `xm.JobRequirements` defaults `replicas` to 1 and nothing here ever set it.
# That is right for a TPU trainer -- one work unit drives the whole slice -- and
# wrong for an embarrassingly-parallel CPU job, where the task count IS the
# parallelism: a 4,800 core-hour corpus at one task is 200 days.
#
# Kept as a separate flag rather than folded into --tpu_type because they are
# different axes: --tpu_type sizes ONE task, this says how many of them. Note
# `replicas` is NOT validated by JobRequirements (0 silently becomes 1, -1 and
# 1.5 survive intact), so the int() conversion below is the only guard there is.
# 1 (default) omits the kwarg entirely, so no existing job changes shape.
_REPLICAS = flags.DEFINE_integer(
    'replicas', 1,
    'Number of Borg tasks for this job. 1 (default) emits no `replicas` '
    'requirement at all, which is exactly what every job did before this flag '
    'existed. Each task gets the full --tpu_type/--ram_gib requirement, and a '
    'CPU job trickles in as capacity appears rather than gang-scheduling. '
    'Remember Borg caps tasks per user per priority tier per CELL (2000 at '
    'priority 0, 400 at priority 25), so a wider run needs more than one job.'
)
# Autopilot resizes a job's CPU/RAM from observed usage. XManager turns it ON
# for every job (the generated BCL calls `autopilot_utils.enable_autopilot`),
# which is right for a server whose footprint is unknown and wrong for a batch
# job whose footprint was measured -- because the resize is not a cap, it is a
# REQUEST. Measured on a 1-cpu / 2 GiB CPU job: enabling Autopilot rewrites the
# ask to `min_milligcu = 1024, max_milligcu = 40000`, and the scheduler then
# looks for a machine with 40 cores free at best-effort priority. It does not
# find one, and the job sits in DISABLED / WAITING_FOR_AUTOPILOT reporting
# "Not enough best effort resources available in the cell" -- a message that
# reads like a capacity problem in the cell and is actually the size of the ask.
#
# With `--autopilot=false` the requirements block is exactly what was asked for
# (`milligcu = 1000`, `ram = 2147483648`) and the BCL calls disable_autopilot.
#
# Default None = leave XManager's behaviour alone, so no existing job changes.
# Turn it off when the per-task footprint is known; keep it on when it is not,
# because then Autopilot is what stops a 2000-task job from OOM-killing itself.
_AUTOPILOT = flags.DEFINE_bool(
    'autopilot', None,
    'Whether Borg Autopilot may resize this job. Unset (default) keeps '
    "XManager's own behaviour, which is ON. Pass --noautopilot for a batch job "
    'whose per-task CPU/RAM is already measured: Autopilot widens the request '
    'to a range (measured: 1 cpu -> min 1024 / max 40000 milligcu) and a '
    'best-effort job then cannot be placed at all.'
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
# TRAINING RESUME (ELT): distinct from --load_from. load_from is READ-ONLY (eval
# / warm-start) and makes workdir=load_from, so an orbax manager that prunes
# deletes the very checkpoints it resumed from. A training run that keeps
# checkpointing must resume via a READ/WRITE SPLIT: read the OLD run's workdir,
# write the NEW car's own $CHECKPOINT_BUCKET, self-clearing once it saves its
# first checkpoint. These two feed config.train.restart_from via
# configs/load_config.py::_apply_restart_from_env (proven by :restart_from_test
# / RESTART_FROM_OK). Ported from the ELT checkout copy 2026-09-10: the shared
# launcher forwarded --restart_from/--restart_step to the trainer as UNDECLARED
# flags (silently dropped), so every ELT resume through this launcher
# cold-started (xid 288423570, and sibling 288408898).
_RESTART_FROM = flags.DEFINE_string(
    'restart_from', '',
    'WORKDIR of a previous run to resume from -- the directory CONTAINING '
    '`checkpoints/`, never the leaf step dir. Exported as $ELT_RESTART_FROM. '
    'Requires --restart_step. Writes still go to $CHECKPOINT_BUCKET.'
)
_RESTART_STEP = flags.DEFINE_string(
    'restart_step', '',
    'Step to resume from, named explicitly. Exported as $ELT_RESTART_STEP. '
    'Mandatory with --restart_from: an unnamed step resolves silently and a '
    'wrong resume point reads as training instability, not a launch error.'
)
_CELL_PREFER = flags.DEFINE_string(
    'cell_prefer', 'auto',
    "Where the scheduler may place this job, best-first. 'auto' (default) uses "
    "the storage-aware tiers in _CELL_TIERS: the mirrored-data metros first, "
    "other group-quota metros next, and cells with no team storage last. "
    "'off' disables the constraint entirely. A comma-separated cell list "
    "overrides the tiers. IGNORED when --cell pins one cell explicitly."
)
_CELL = flags.DEFINE_string(
    'cell', '',
    'Borg cell to pin the job to (e.g. nz, pa, go), or "viglobal" to let XBorg '
    'choose. GQM clears prices PER CELL, so pinning to a cheap cell is often '
    'the difference between running and sitting in the '
    'TRIGGERED_LIMIT_ORDER bucket. Empty = let the allocator decide.'
)

def _warn_if_work_units_still_active(experiment, xid: str) -> None:
    """A resume APPENDS a work unit; it does not stop the ones already there.

    Two live work units on one XID share `$CHECKPOINT_BUCKET`, because the
    prefix is derived from the experiment id. That is two trainers writing one
    checkpoint series and two mirrors appending to one log -- exactly the
    collision this launcher spends real effort avoiding everywhere else.

    Only a WARNING: an already-terminal experiment is the normal case, the API
    may decline to answer, and refusing a legitimate resume would be worse than
    saying so loudly.
    """
    try:
        active = [
            wu for wu in experiment.get_work_units()
            if str(getattr(getattr(wu, "status", None), "name", "")).upper()
            in ("RUNNING", "PENDING", "SCHEDULING")
        ]
    except Exception as exc:  # noqa: BLE001 - cannot tell is not a failure
        print(f"[resume] could not check XID {xid} for live work units "
              f"({type(exc).__name__}: {exc}); proceeding.")
        return
    if not active:
        return
    print(
        f"\n[resume] WARNING: XID {xid} still has {len(active)} work unit(s) in a "
        f"live state.\n"
        "[resume]   A resume APPENDS a work unit and does not stop those. Both "
        "would write\n"
        "[resume]   the SAME checkpoint prefix (it is derived from the XID) and "
        "the same\n"
        "[resume]   log files -- two trainers on one series. Stop the old one "
        f"first:\n[resume]       tpu stop {xid}\n"
    )


def _preflight_resume_config(bucket_cp_path: str, xid: str) -> None:
    """Abort a `--resume_xid` whose config describes a different model.

    Best effort by design: this runs on the workstation, where the project's
    checkpoint helpers may not be importable and the bucket may not be readable.
    Any of that means "cannot tell", and cannot-tell must not block a launch --
    the job re-checks the same thing at startup. It only ever aborts on a
    DIFFERENCE IT CAN PROVE.
    """
    import sys

    # A MISSING CONFIG FILE IS A DIFFERENCE WE CAN PROVE, not a cannot-tell.
    # The generic except below treats every failure as "skip", which once let a
    # `--config=configs/x_config.yml` (should be the bare mode `x`) double-wrap
    # into a path that cannot exist and sail through to a job that died at
    # startup. --config is already normalised to the bare mode in main(), so the
    # file the launcher will actually load is configs/<mode>_config.yml; if that
    # is absent, refuse now rather than package a doomed job.
    _cfg_file = f"configs/{_CONFIG.value}_config.yml"
    if not os.path.exists(_cfg_file) and not os.path.exists(
            f"configs/{_CONFIG.value}_config.yaml"):
        raise SystemExit(
            f"\n=== REFUSING TO RESUME XID {xid} ===\n"
            f"config file not found: {_cfg_file}\n"
            f"--config must be a bare mode name (e.g. remote_run); the launcher "
            f"wraps it into configs/<mode>_config.yml.\n"
            f"\nNothing was packaged or queued.\n"
        )

    try:
        sys.path.insert(0, os.getcwd())
        from configs import load_config
        from utils.ckpt_util import (
            assert_config_matches_checkpoint,
            join_path,
            latest_checkpoint,
        )

        cfg = load_config.get_config(_CONFIG.value)
        resume_from = latest_checkpoint(join_path(bucket_cp_path, "checkpoints"))
    except Exception as exc:  # noqa: BLE001 - cannot tell is not a failure
        print(f"[resume] pre-flight config check skipped ({type(exc).__name__}: {exc})")
        return

    if not resume_from:
        print(f"[resume] no complete checkpoint under {bucket_cp_path}/checkpoints yet; "
              "the job will start fresh on this prefix.")
        return

    try:
        assert_config_matches_checkpoint(
            resume_from, cfg, context=f" (--resume_xid={xid})"
        )
    except Exception as exc:  # noqa: BLE001 - this one IS the answer
        raise SystemExit(
            f"\n=== REFUSING TO RESUME XID {xid} ===\n{exc}\n"
            f"\nNothing was packaged or queued.\n"
        )
    print(f"[resume] config matches {resume_from}")


def main(argv) -> None:
    # NORMALISE --config to the bare <mode>. The contract is a short mode name
    # (`remote_run`); load_config and the launcher then wrap it into
    # `configs/<mode>_config.yml`. Passing the full path instead --
    # `--config=configs/abl_config.yml` -- makes every call site double-wrap it
    # into `configs/configs/abl_config.yml_config.yml`, a file that cannot exist,
    # and the job dies at startup with "Could not locate ...". Strip the prefix
    # and suffix so a full path is accepted as if the mode had been passed.
    _raw_config = _CONFIG.value
    _mode = _raw_config.strip()
    if _mode.startswith('configs/'):
        _mode = _mode[len('configs/'):]
    for _suffix in ('_config.yml', '_config.yaml', '.yml', '.yaml'):
        if _mode.endswith(_suffix):
            _mode = _mode[:-len(_suffix)]
            break
    if _mode != _raw_config:
        print(f"[config] normalised --config={_raw_config!r} -> {_mode!r} "
              "(pass the bare mode name; the launcher adds configs/…_config.yml)")
        flags.FLAGS.config = _mode

    exp_name = _EXP_NAME.value
    # --- Auto-Load WandB name or fallbacks ---
    cfg = None
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
    # ONCE, not per tpu_type: `--tpu_type=a,b` builds one executor per entry and
    # the answer is a property of the DATASET, so measuring inside that loop
    # would shell out to `fileutil du` once per candidate for one identical
    # number.
    tmp_ram_fs_gib = _resolve_tmp_ram_fs_gib(cfg)
    for arg in argv[1:]:
        if arg.startswith('--config.wandb.notes='):
            exp_name = arg.split('=', 1)[1]
        elif arg.startswith('--config.wandb.run_name='):
            exp_name = arg.split('=', 1)[1]
            
    # Mark whose job this is in the XManager UI. The job registry already
    # separates operators on this workstation (TPU_JOBS_FILE), but XM shows
    # one shared account, so without this a collaborator's experiment is
    # indistinguishable from the owner's in the only view other people see.
    _title_prefix = os.environ.get("TPU_JOB_NAME_PREFIX", "")
    if _title_prefix and not exp_name.startswith(_title_prefix):
        exp_name = f"{_title_prefix}{exp_name}"

    # --- Research Hub 归属提示:跳过它,不要等满 25 分钟 -------------------
    # 【2026-08-21 由 lyy 明确授权改这一处。备份:
    #   ~/lyy-work/.bak_xm_launcher.py.20260821_112107】
    #
    # 2026-08-21 08:11 起,本工作站每一次 create_experiment() 都会触发 Research
    # Hub 的 effort 归属交互提示。alloc 'fr-dna-grand-challenge-team-resource'
    # 没有链到本账号任何 active effort,提示因此每次都弹。
    #
    # 它在无人值守场景里**无法被回答**,两条独立原因:
    #   * 它不读 stdin,直接开 /dev/tty(//depot/google3/pyglib/promptutil.py:570
    #     tty_raw_input,默认 _SubstituteTtyStdin),所以 `< /dev/null` 无效;
    #   * attribution_urls 没有「空值即跳过」的语义 —— policy.py:480 是
    #     `if description.attribution_urls:`,传 [] 或 '' 都落进 else 分支照弹。
    #
    # 代价是 5 次 × 300 秒 = **最多 25 分钟纯空转**
    #   (_MAXIMUM_PROMPT_LIMIT=5 见 interactive.py:28;
    #    xm_attribution_prompt_timeout_seconds 默认 300 见 policy_flags.py:18)。
    #
    # **关掉 enforcement 不改变归属结果。** 今天每一条投放最后都是无归属的 ——
    # 提示答不上,它等满就放行。所以这一段去掉的是空转,不是记账。
    # policy.py:373-379 读到该 flag 为真就 return True,不发任何 Research Hub
    # RPC、不弹提示,telemetry 里记 skip_reason=DISABLE_FLAG,**不会把开销记到
    # 任何真实 effort 上** —— 那正是不能随手编一个 rh/efforts/NNNN 的理由。
    # 该 flag 是官方 opt-out(policy_flags.py:28 'Disable attribution
    # enforcement for this launch'),ceres_ml 等生产 pipeline 成批在用。
    #
    # 拿到真实 effort 号之后应该换成显式归属,并把这一段删掉。两种写法:
    #     export XM_ATTRIBUTION_URLS=rh/efforts/NNNN     # 无需改本文件
    #     xm_abc.create_experiment(..., attribution_urls=['rh/efforts/NNNN'])
    #
    # 为什么必须先摸一下 xm_abc 的属性:xm_abc 是懒加载包
    # (xm_abc/__init__.py 用 module_lazy_loader,真正的 import 写在
    #  `if typing.TYPE_CHECKING:` 里),不触发导入的话 policy_flags 还没被加载,
    # 这个 flag 名在 flags.FLAGS 里根本不存在。
    # 也正因如此**不能从命令行传**:app.run() 在 main() 之前解析 argv,那时名字
    # 可能还没注册,而未知的 --flag 会穿过 known_only=True 的解析,被下面约
    # 1183 行的转发循环塞给 trainer,把训练进程搞崩。
    #
    # --resume_xid 那条分支不需要处理:xm_abc.get_experiment 只做一次
    # get_experiment RPC,全 google3 里 maybe_enforce_attribution_urls 的唯一
    # 非测试调用点是 launcher.py:433,只挂在新建实验这条路上。
    _ = xm_abc.create_experiment          # 触发懒加载,注册 policy_flags
    _ATTR_OPT_OUT = 'xm_disable_effort_attribution_enforcement'
    if _ATTR_OPT_OUT in flags.FLAGS:
        flags.FLAGS[_ATTR_OPT_OUT].value = True
        print('[launcher] Research Hub 归属强制已关闭(官方 opt-out flag);'
              '本次投放不会因归属提示空转。拿到真实 rh/efforts 号后请改用 '
              'XM_ATTRIBUTION_URLS。')
    else:
        print(f'[launcher] 警告:{_ATTR_OPT_OUT} 未注册,归属提示可能仍会出现'
              f'(最多空转 25 分钟)。请检查 xm_abc 懒加载是否变了。')
    # ----------------------------------------------------------------------

    experiment_context = xm_abc.get_experiment(experiment_id=_RESUME_XID.value) if _RESUME_XID.value else xm_abc.create_experiment(experiment_title=exp_name)
    with experiment_context as experiment:
        
        # --- Codebase Specifics via config.sh ---
        project_name = "unified_project"
        package_mode = "python" # 'python' or 'bazel'
        target_label = "."
        
        pkg_path = target_label.lstrip('/').split(':')[0]
        
        # Locate config.sh ROBUSTLY. This read decides package_mode, which
        # routes bazel_binary (internal Borg) vs python_container (GCP, needs a
        # mapped cloud project). A SILENT read failure here used to default
        # package_mode to "python" -> python_container -> GCP get_project(
        # 'deepmind-dynamic') -> NotImplementedError: No project set, killing
        # every PROD arm at launch. The read failed silently because config.sh
        # was resolved CWD-RELATIVE, and the wrapper's getcwd guard chdir's to
        # $STAGE_WS_ROOT (which has no top-level config.sh) right before launch.
        # Fix: resolve via $TPU_STAGEDIR (exported by the wrapper on BOTH the
        # fresh-build and resume paths) FIRST, then fall back to CWD; and make a
        # failure LOUD instead of defaulting to python.
        _stagedir = os.environ.get("TPU_STAGEDIR", "")
        _config_candidates = []
        if _stagedir:
            _config_candidates.append(os.path.join(_stagedir, "config.sh"))
        _config_candidates.append("config.sh")
        if pkg_path:
            _config_candidates.append(f"{pkg_path}/config.sh")
        _config_path = next((c for c in _config_candidates if os.path.exists(c)), None)
        if _config_path is None:
            raise SystemExit(
                "[launcher] FATAL: config.sh not found in any of "
                f"{_config_candidates} (TPU_STAGEDIR={_stagedir!r}, cwd={os.getcwd()!r}). "
                "Refusing to guess package_mode: defaulting to 'python' would route "
                "to python_container/GCP and die with 'No project set for pool_name: "
                "deepmind-dynamic'. Ensure the wrapper staged config.sh and exported "
                "TPU_STAGEDIR."
            )
        with open(_config_path, "r") as f:
            for line in f:
                if line.startswith("export PROJECT_NAME="):
                    project_name = line.split("=")[1].strip().strip('"').strip("'")
                elif line.startswith("export PACKAGE_MODE="):
                    package_mode = line.split("=")[1].strip().strip('"').strip("'")
                elif line.startswith("export TARGET_LABEL="):
                    target_label = line.split("=")[1].strip().strip('"').strip("'")
                    pkg_path = target_label.lstrip('/').split(':')[0]
        print(f"[launcher] config.sh={_config_path} -> package_mode={package_mode!r} "
              f"target_label={target_label!r} project_name={project_name!r}")

        print("[trace] A/executors-init", flush=True)   # TEMP probe 2026-08-31 chipwatch; remove after resume_xid diagnosed
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
        # NVIDIA GPUs. xm.ResourceType is CASE-INSENSITIVE and the kwarg name IS
        # the lowercase enum name, so JobRequirements(h100=8) / (b200=8) /
        # (gb200=8) work directly -- the map is identity, present so the GPU
        # branch below can recognise these arch tokens and so a typo gets a
        # clean error instead of a cryptic ResourceType KeyError at submit.
        # NVLINK_DOMAIN is the device_group size from the platform GCL
        # (platforms/accelerator_metadata/platforms/*.gcl): the largest slice
        # that is fully NVLink-connected. Above it, chips talk over network
        # RDMA, so a bigger single-job ask is legal but not faster for
        # comms-bound work -- we warn, not block.
        GPU_MAP = {
            "a100": "a100",          # 46  (40 GiB)
            "a100_80gib": "a100_80gib",  # 66
            "h100": "h100",          # 70
            "h200": "h200",          # 86
            "b200": "b200",          # 87
            "b300": "b300",          # 112
            "gb200": "gb200",        # 89
            "gb300": "gb300",        # 100
        }
        NVLINK_DOMAIN = {
            "a100": 16, "a100_80gib": 8, "h100": 8, "h200": 8,
            "b200": 8, "b300": 8, "gb200": 72, "gb300": 72,
        }
        FISH_MAP.update(GPU_MAP)
        
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

        # Megacore (v4/pufferfish ONLY). v4 exposes 2 TensorCores per chip, so
        # JAX sees 2 devices/chip and jax.make_mesh() aborts at init with
        # "Creating meshes for TPU >v3 requires one device per chip (megacore
        # mode)" (observed: XIDs 288531077/288568565 on v4-256 died here while
        # the identical code ran fine to 28k steps on v6p, which is 1 core/chip).
        # deepsea_chip_config_name=megacore_dense fuses the two cores into one
        # logical device so device_count halves and the mesh builds -- the exact
        # fix paligemma's launcher uses (//third_party/py/big_vision/launch.py
        # :290, `if device_type == "pf": exe_args[...]="megacore_dense"`).
        # megacore is pufferfish-only (ghostfish/-lite and viperfish reject it),
        # and the flag is one job-wide executable arg (not per-executor), so we
        # only set it for a v4-ONLY job and warn on a mixed Fallback.
        _tpu_res_names = []

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
                elif arch_lower in GPU_MAP:
                    # NVIDIA GPUs are NOT a torus: the chip count passes through
                    # UNCHANGED as a scalar (JobRequirements(h100=8) -> a 1-D
                    # topology of 8, one task). No TPU min-slice rule applies
                    # (those are torus/pod fragmentation limits). We only WARN
                    # when the ask exceeds the card's NVLink domain, because
                    # past that boundary the extra chips talk over network RDMA
                    # rather than NVLink -- legal, but not faster for
                    # communication-bound work, and multi-host GPU coordination
                    # (torchrun/NCCL) is the caller's responsibility, not the
                    # JAX-coordination path this launcher injects for TPUs.
                    domain = NVLINK_DOMAIN.get(arch_lower)
                    if domain and num_cores > domain:
                        print(f"[gpu] WARNING: requested {arch}-{num_cores} exceeds the "
                              f"{domain}-GPU NVLink domain for {arch}. Chips beyond "
                              f"{domain} communicate over network RDMA, not NVLink; "
                              f"comms-bound work will not scale linearly, and you must "
                              f"handle multi-host GPU coordination yourself.")
                    # cores stays the raw integer string; no torus remap.
                
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
            # int(): a fractional GiB must still lower to a whole number of
            # bytes, or the BCL carries a float and Borg rejects it.
            req_kwargs['tmp_ram_fs'] = int(tmp_ram_fs_gib * xm.GiB)
            if _RAM_GIB.value > 0:
                req_kwargs['ram'] = int(_RAM_GIB.value * xm.GiB)
            # `replicas` is the task count. Emitted only above 1, so a job that
            # does not ask for it keeps the exact requirements block it had
            # before this flag existed. int() is deliberate: JobRequirements
            # does NOT validate this kwarg (measured -- 0 becomes 1, -1 and 1.5
            # pass straight through into the BCL), so a float or a string would
            # travel all the way to Borg unexamined.
            if _REPLICAS.value > 1:
                req_kwargs['replicas'] = int(_REPLICAS.value)
            if alloc_str:
                req_kwargs['allocator'] = alloc_str
            if _CELL.value:
                # An explicit cell wins outright: pin it, and do not also send
                # spatial-flexibility constraints, which are only read when the
                # location is 'viglobal' and would be silently ignored anyway.
                req_kwargs['location'] = _CELL.value
            else:
                # NO EXPLICIT CELL -> let XBorg choose, but only from cells we
                # have storage in, best tier first.
                #
                # `allowed_locations` is read ONLY when location == 'viglobal'
                # ("Unused if the job location is not viglobal",
                # xm/resources.py), so the two have to be set together -- which
                # is also what rs.Location does internally
                # (resource_selector/constraints.py::add_viglobal_requirement).
                #
                # FAIL-OPEN on purpose. The xmanager constraint classes cannot
                # be imported or exercised from a workstation, so this path is
                # only ever executed for real at submit time. If anything about
                # it is wrong, the job must still launch the way it did before
                # this feature existed -- placement is an optimisation, and a
                # launcher that refuses to submit is worse than one that places
                # a job suboptimally.
                cells = _preferred_cells(_CELL_PREFER.value)
                if cells:
                    try:
                        from xmanager.xm_abc import constraints as _abc_constraints

                        req_kwargs['location'] = 'viglobal'
                        req_kwargs['spatial_flexibility_constraints'] = (
                            _abc_constraints.SpatialFlexibilityConstraints(
                                allowed_locations=list(cells)
                            )
                        )
                        print(f"[locality] no --cell given; allowing {len(cells)} cell(s) "
                              f"via viglobal (--cell_prefer={_CELL_PREFER.value}):")
                        _describe_cell_choice(list(cells))
                    except Exception as exc:  # noqa: BLE001 - placement must never block a launch
                        req_kwargs.pop('spatial_flexibility_constraints', None)
                        req_kwargs.pop('location', None)
                        print(f"[locality] could not apply cell preference ({exc}); "
                              "letting the allocator choose, as before.")
            tier_val = None
            for a in argv[1:]:
                if a.startswith('--tier='):
                    tier_val = a.split('=', 1)[1].upper()
                    if tier_val == 'PROD':
                        req_kwargs['service_tier'] = xm.ServiceTier.PROD
                    elif tier_val == 'BATCH':
                        req_kwargs['service_tier'] = xm.ServiceTier.BATCH
                    # A NUMERIC tier is a raw Borg priority. JobRequirements
                    # accepts `priority=` directly and DERIVES the service tier
                    # from it, so this one branch reaches every band -- p0 and
                    # p25 (free, charged to the USER) as well as the two named
                    # tiers -- with no new flag and no enum table to keep in
                    # sync. `priority=` and `service_tier=` are mutually
                    # exclusive (resources.py raises if both are given), which
                    # is why this is elif and not a second assignment.
                    #
                    # Two traps worth naming, because both are counter-intuitive
                    # and both cost money:
                    #  * BATCH (p100) is a PAYING best-effort tier and it charges
                    #    the GROUP, not you. It is not the free option; p0/p25
                    #    are (borg.py: at priority <= 25 the BCL carries
                    #    `accounting.charged_user = "<user>"`).
                    #  * The task-per-cell cap is per PRIORITY TIER: 2000 at p0
                    #    but only 400 at p25, so the lower number is the roomier
                    #    one. See //production/borg/usermaps/setup/
                    #    default_allocation.gcl and //borg/control/g3doc/limits.md.
                    #
                    # WITHOUT this branch `--tier=0` matched neither string,
                    # left service_tier unset, and silently took the
                    # JobRequirements default -- which is PROD. Asking for the
                    # free tier and being given the most expensive one is the
                    # worst possible failure for a flag like this.
                    elif tier_val.isdigit():
                        req_kwargs['priority'] = int(tier_val)
                    else:
                        raise ValueError(
                            f"--tier={tier_val!r} is neither PROD, BATCH, nor a "
                            "numeric Borg priority. Refusing to submit: an "
                            "unrecognised tier used to fall through to PROD.")
            job_requirements = xm.JobRequirements(**req_kwargs)
            
            # Borg tracks true task failures separately from evictions. The
            # public BorgScheduling object exposes only the former; the finite
            # eviction budget is added below through one narrow Borg override.
            # See //depot/google3/third_party/py/xmanager/xm_abc/executors.py
            # (class BorgScheduling) and go/borg-configure-schedule.
            scheduling = xm_abc.BorgScheduling(
                max_task_failures=_MAX_TASK_FAILURES.value,
                max_per_task_failures=_MAX_PER_TASK_FAILURES.value,
                task_failure_credit_period=_FAILURE_CREDIT_PERIOD.value,
            )

            # Only construct AutopilotParams when the flag was actually given.
            # An all-default AutopilotParams is not inert: the executor treats
            # `enabled=None` plus any other field as "parameters you set that I
            # will ignore" and prints a warning, and passing the object at all
            # is a change in shape for every existing job.
            borg_kwargs = {}
            eviction_overrides = _borg_overrides_for_eviction_budget(
                _MAX_TASK_EVICTIONS.value)
            if eviction_overrides is not None:
                borg_kwargs['borg_overrides'] = eviction_overrides
            if _AUTOPILOT.value is not None:
                borg_kwargs['autopilot_params'] = xm_abc.AutopilotParams(
                    enabled=_AUTOPILOT.value)
            if package_mode == "bazel":
                executor = xm_abc.Borg(
                    requirements=job_requirements,
                    scheduling=scheduling,
                    # Let colleagues (and future you) read this job's logs
                    # without an ACL dance.
                    logs_read_access_roles=['all'],
                    **borg_kwargs,
                )
            else:
                executor = xm_abc.Gcp(requirements=job_requirements)
            executors.append(executor)
            # Any TPU accelerator in the request means the job is multi-task and
            # needs the JAX coordination flags injected below. GPUs are NOT
            # TPU jobs: FISH_MAP now also carries the GPU families, so match
            # against the TPU codenames ONLY (GPU_MAP is the exclusion set),
            # or the GPU path would wrongly get the TPU JAX-coordination flags.
            _is_gpu = res_name in GPU_MAP
            if not _is_gpu and (res_name in FISH_MAP.values()
                                or res_name.startswith('tpu')):
                is_tpu_job = True
            _tpu_res_names.append(res_name)

        print("[trace] B/tpu-loop-done", flush=True)   # TEMP probe 2026-08-31 chipwatch; remove after resume_xid diagnosed
        final_executor = xm.Fallback(executors) if len(executors) > 1 else executors[0]
    
        print("[trace] C/got-xid", flush=True)   # TEMP probe 2026-08-31 chipwatch; remove after resume_xid diagnosed
        xid = experiment.experiment_id
        import time
        time_str = time.strftime("%Y%m%d_%H%M%S")

        # SANITISE THE NAME BEFORE IT BECOMES A PATH. `-n` is free text and it
        # lands verbatim in a CNS directory name, where Colossus treats
        # * ? [ ] as glob metacharacters and REJECTS them: a title containing
        # brackets makes the job's own `ensure_dir` fail with INVALID_ARGUMENT.
        #
        # XID 277172543 died exactly this way, and the shape of that failure is
        # why this is worth guarding rather than remembering: it ran for twelve
        # minutes first, and it produced NO log at all, because the log mirror
        # is itself created under the same directory and its failure path is a
        # bare `except: return None`. So the symptom was a silent burn of a
        # v7-16 allocation with an empty status message -- indistinguishable at
        # a glance from an infrastructure fault. It was relaunched by hand with
        # the brackets removed, which fixed that job and left the trap armed.
        #
        # Replace rather than strip, so two titles differing only in punctuation
        # cannot collide on one checkpoint prefix.
        safe_exp_name = re.sub(r"[\*\?\[\]]", "_", exp_name)
        if safe_exp_name != exp_name:
            print(f"[launcher] NOTE: the experiment title contains characters CNS reserves "
                  f"for globbing (* ? [ ]); the checkpoint directory will use "
                  f"{safe_exp_name!r}. The experiment title itself is unchanged.")
        folder_name = f"xid_{xid}_{time_str}_{safe_exp_name}"

        import json
        import fcntl
        mapping_file = os.environ.get("TPU_JOBS_FILE") or os.path.expanduser("~/.tpu_jobs.json")

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

        print("[trace] D/before-resume-block", flush=True)   # TEMP probe 2026-08-31 chipwatch; remove after resume_xid diagnosed
        if _RESUME_XID.value:
            # LOOK IN THE ARCHIVE TOO. `tpu clear` advertises itself as archiving
            # rather than deleting -- it moves entries to ~/.tpu_jobs_legacy.json --
            # but resume only consulted the live registry, so clearing a finished
            # run quietly made it un-resumable. The failure was not even a clear
            # message: it fell through to the long-dead ~/xm_job_to_bucket/ path
            # and raised FileNotFoundError on a file nothing has written since
            # 2026-07-26.
            print("[trace] D1/enter-branch", flush=True)   # TEMP probe 2026-08-31 chipwatch
            want = str(_RESUME_XID.value)
            bucket_cp_path = ""
            for source in (read_mapping(), _read_legacy_mapping()):
                if want in source:
                    bucket_cp_path = source[want].get("bucket_cp_path", "")
                    if bucket_cp_path:
                        break
            print("[trace] D2/mapping-looked-up", flush=True)   # TEMP probe 2026-08-31 chipwatch
            if not bucket_cp_path:
                # Last resort: the pre-2026-07-26 one-file-per-xid layout.
                legacy_file = os.path.join(
                    os.path.expanduser("~/xm_job_to_bucket"), want)
                if not os.path.exists(legacy_file):
                    raise SystemExit(
                        f"--resume_xid={want}: no checkpoint bucket recorded for that "
                        f"experiment.\nLooked in {os.environ.get('TPU_JOBS_FILE') or '~/.tpu_jobs.json'}, "
                        f"~/.tpu_jobs_legacy.json and {legacy_file}.\n"
                        f"Pass --bucket=<its bucket_cp_path> explicitly if you know it.")
                with open(legacy_file, "r") as f:
                    bucket_cp_path = f.read().strip()
            print("[trace] D3/have-bucket", flush=True)   # TEMP probe 2026-08-31 chipwatch
            vm_workdir = f"/tmp/eqr_log/resume_{xid}_{time_str}_{project_name}_{exp_name}"
            # PRE-FLIGHT: does the config we are about to package actually
            # describe the checkpoint we are about to resume?
            #
            # A resume packages the CURRENT checkout and reads the project's
            # single shared run config. That file is overwritten by every
            # launch, so resuming a run from days ago routinely ships the wrong
            # experiment's config -- two 150k-step sudoku runs were resumed
            # against a maze config and died on arrival, after ten minutes of
            # packaging and queueing. Checking here costs one small read of the
            # checkpoint's `extra.json` and fails in seconds instead.
            print("[trace] D4/before-preflight-cfg", flush=True)   # TEMP probe 2026-08-31 chipwatch
            _preflight_resume_config(bucket_cp_path, xid)
            # And a resume does not stop what is already running on this XID.
            print("[trace] D5/before-warn-active", flush=True)   # TEMP probe 2026-08-31 chipwatch
            _warn_if_work_units_still_active(experiment, xid)
        else:
            bucket_cp_path = f"{_local_bucket()}/logs/{project_name}/{folder_name}"
            vm_workdir = f"/tmp/eqr_log/{folder_name}"
            
        # The registry entry is written AFTER `experiment.add(job)`, at the end
        # of this function -- see the comment there. Building it here, where the
        # values are in scope, keeps that move a pure reordering.
        print("[trace] E/resume-block-done", flush=True)   # TEMP probe 2026-08-31 chipwatch; remove after resume_xid diagnosed
        registry_entry = {
            "bucket_cp_path": bucket_cp_path,
            "logdir": os.environ.get("TPU_LOGDIR", ""),
            "stagedir": os.environ.get("TPU_STAGEDIR", ""),
            "exp_name": exp_name,
            "tpu_type": _TPU_TYPE.value,
            # Recorded so `tpu check` can tell "preempted, will retry" from
            # "preempted, restart budget spent".
            "max_task_failures": _MAX_TASK_FAILURES.value,
            "max_task_evictions": _MAX_TASK_EVICTIONS.value,
        }
    
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
            # v4/pufferfish megacore fix (see note at `_tpu_res_names =` above).
            # Set megacore_dense ONLY when the job is v4-only; a mixed Fallback
            # cannot carry a per-executor value, so warn rather than break the
            # non-v4 arms.
            _v4_res = [r for r in _tpu_res_names if r == 'pufferfish']
            _distinct_res = set(_tpu_res_names)
            if _v4_res and len(_distinct_res) == 1:
                executable_args['deepsea_chip_config_name'] = 'megacore_dense'
                print('[launcher] v4/pufferfish: deepsea_chip_config_name='
                      'megacore_dense (fuse 2 cores/chip -> 1 device so '
                      'jax.make_mesh() works).')
            elif _v4_res:
                print('[launcher] WARNING: v4/pufferfish is mixed with other '
                      f'archs ({sorted(_distinct_res)}); deepsea_chip_config_name '
                      'is one job-wide flag and cannot be set per-executor, so '
                      'the v4 arm would hit the megacore mesh assertion. Enqueue '
                      'v4 as its own single-arch job to get megacore_dense.')
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
                                   '--restart_from=', '--restart_step=',
                                   '--borg_max_task_failures=', '--borg_max_per_task_failures=',
                                   '--borg_max_task_evictions=',
                                   '--tmp_ram_fs_gib=', '--ram_gib=', '--replicas=',
                                   '--autopilot=', '--noautopilot', '--autopilot')):
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
        print("[trace] F/env-vars", flush=True)   # TEMP probe 2026-08-31 chipwatch; remove after resume_xid diagnosed
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
        # TRAINING RESUME env delivery. Fail closed on a half-specified resume:
        # a path with no step would otherwise reach the job and raise there,
        # burning a build + a slice. The trainer's _apply_restart_from_env reads
        # exactly these two names.
        if _RESTART_FROM.value or _RESTART_STEP.value:
            if not (_RESTART_FROM.value and _RESTART_STEP.value):
                raise ValueError(
                    '--restart_from and --restart_step must be given TOGETHER; '
                    f'got restart_from={_RESTART_FROM.value!r} '
                    f'restart_step={_RESTART_STEP.value!r}')
            job_env_vars['ELT_RESTART_FROM'] = _RESTART_FROM.value
            job_env_vars['ELT_RESTART_STEP'] = _RESTART_STEP.value
        # Where the job should persist its own checkpoints. workdir lives on
        # the task's local disk, which is wiped on every Borg task restart, so
        # a durable copy has to go to GCS for a restart to be able to resume.
        job_env_vars['CHECKPOINT_BUCKET'] = bucket_cp_path

        # RoboTwin close-loop eval: tell the job where to mirror checkpoints.
        # The job reads these in `utils/gcs_publish.py`; their ABSENCE is what
        # disables publishing, so a non-RoboTwin run cannot upload by accident.
        if _should_publish_to_gcs():
            job_env_vars['ROBOTWIN_EVAL_BUCKET'] = _ROBOTWIN_EVAL_BUCKET
            job_env_vars['ROBOTWIN_EVAL_XID'] = str(xid)
            print(f'[gcs-publish] enabled: {_ROBOTWIN_EVAL_BUCKET}/runs/{xid}')
        

        print("[trace] G/before-bazel-package", flush=True)   # TEMP probe 2026-08-31 chipwatch; remove after resume_xid diagnosed
        if package_mode == "bazel":
            # BASE build flags for every bazel job. A GPU job additionally
            # needs CUDA compiled IN: torch's CUDA kernels are `if_cuda`-gated
            # in //third_party/py/torch, and xmanager's apply_default_bazel_args
            # does NOT auto-add --config=cuda from the accelerator -- it must be
            # passed explicitly (verified in xm_abc/packaging/bazel_args.py).
            # Without it a torch/GPU bazel binary builds CPU-ONLY and reports
            # torch.cuda.device_count()==0 at runtime -- the torch twin of the
            # JAX tpu_support trap (a CPU-only build that does not say so).
            #
            # xm_abc.bazel_args.gpu(<resource>) returns exactly what xmanager
            # itself uses for a GPU job: --config=cuda, --define=cuda_compress=1,
            # the per-SM enables (h100 -> sm90), and the accelerator FDO/opt
            # flags. This is the bazel/Borg counterpart of the python_container
            # base_image('pytorch') branch below (which only covers the GCP
            # path). TPU and CPU jobs are untouched: the GPU flags are appended
            # ONLY when res_name is a GPU arch. Fail-open -- a lookup failure
            # falls back to a bare --config=cuda, which still yields CUDA torch,
            # because a launcher that refuses to submit is worse than one that
            # builds with slightly coarser flags.
            _bazel_args = ["--define=PYTYPE=FALSE", "--norun_validations"]
            if res_name in GPU_MAP:
                try:
                    _gpu_res = xm.ResourceType[res_name.upper()]
                    _gpu_flags = tuple(xm_abc.bazel_args.gpu(_gpu_res))
                except Exception as _e:  # noqa: BLE001 - never block a launch on flag lookup
                    _gpu_flags = ("--config=cuda", "--define=cuda_compress=1")
                    print(f"[gpu] bazel_args.gpu({res_name!r}) unavailable "
                          f"({type(_e).__name__}: {_e}); using bare {_gpu_flags}.")
                for _f in _gpu_flags:
                    if _f not in _bazel_args:
                        _bazel_args.append(_f)
                print(f"[gpu] bazel CUDA build flags for {res_name}: {_gpu_flags}")
            (executable,) = experiment.package(
                [xm.bazel_binary(
                    label=target_label,
                    bazel_args=_bazel_args,
                    executor_spec=final_executor.Spec(),
                    args=executable_args,
                    env_vars=job_env_vars,
                )]
            )
        else: # python mode default
            base_image_accel = executors[0].requirements.accelerator
            # Pick the container framework from the accelerator: a GPU job needs
            # the CUDA PyTorch image (framework_defaults.base_image('pytorch',
            # gpu) -> gcr.io/deeplearning-platform-release/pytorch-gpu...),
            # while TPU/JAX keeps the jax image. Hardcoding 'jax' shipped a
            # JAX-only image to a torch-on-GPU job. `is_tpu_job` is already False
            # for GPUs (so no JaxFlags were injected); this makes the IMAGE match
            # too. Override with FRAMEWORK=... in config.sh if a GPU job really
            # wants JAX (jax-on-GPU) or vice versa.
            _fw = os.environ.get('FRAMEWORK', '')
            if not _fw:
                _fw = 'pytorch' if (not is_tpu_job and res_name in GPU_MAP) else 'jax'
            print(f'[launcher] python_container framework={_fw!r} '
                  f'accel={base_image_accel}')
            (executable,) = experiment.package(
                [xm.python_container(
                    path='.',
                    base_image=framework_defaults.base_image(_fw, base_image_accel),
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
        print("[trace] H/before-experiment-add", flush=True)   # TEMP probe 2026-08-31 chipwatch; remove after resume_xid diagnosed
        job = xm.Job(executable, final_executor, args=executable_args, name=_JOB_NAME)
        experiment.add(job)

        # REGISTER ONLY ONCE THE WORK UNIT EXISTS. This used to run ~150 lines
        # earlier, right after the bucket path was computed, which made the
        # entry a PRE-registration: written before `experiment.package()` (the
        # slow part, 1-4 minutes) and before `experiment.add(job)`.
        #
        # That is a real failure mode, not a theoretical one. The launcher runs
        # as a ~3.75 GB PAR executed straight off BinFS, and `binfsd` panics on
        # a schedule (`image_cache.go` logs "Failed to stat" during cache
        # eviction and then dies -- a missing `continue`). When it restarts,
        # /google/bin is remounted and every process holding an mmap of it takes
        # SIGBUS on its next page fault. SIGBUS is not a Python exception: there
        # is no traceback, no `finally`, and nothing cleans up. Six launches on
        # 2026-08-04 left an XID in the registry with no experiment behind it,
        # each carrying EXACTLY the six keys written above, and `tpu check`
        # showed them forever as "unknown ... No WorkUnits (config error?)".
        #
        # Nothing between the old site and here reads the registry, so moving
        # the write is a pure reordering. The invariant it buys: an entry exists
        # only if a work unit was actually added.
        update_mapping(xid, registry_entry)

        # Publish the code snapshot LAST, for the same reason the registry
        # write moved here: only a launch that actually produced a work unit
        # should leave anything behind. This runs on the workstation (the only
        # place that can read CitC and reach GCS) and cannot fail the launch.
        if _should_publish_to_gcs():
            _publish_stagedir(str(xid), {
                'xid': str(xid),
                'exp_name': exp_name,
                'robotwin_task': _robotwin_task_from_yaml(),
                'config': _CONFIG.value,
                'cns_checkpoint_path': bucket_cp_path,
                'stagedir': os.environ.get('TPU_STAGEDIR', ''),
                'tpu_type': _TPU_TYPE.value,
                'published_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
            })


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
