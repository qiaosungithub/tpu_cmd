"""Unit tests for the local-queue CLI. Drives the subcommands through their flag
objects against a temp queue file; no RPC (status' live probe is not exercised)."""

import contextlib
import io
import os
import tempfile
import time
import unittest
from unittest import mock

from absl import flags

from google3.experimental.users.qiaos.tpu_utils import queue_cli as Q
from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R


def setUpModule():
  # The subcommands read flag .value directly; without app.run the flags are
  # "unparsed" and absl refuses access. Mark them parsed once for the module.
  flags.FLAGS.mark_as_parsed()


class ParseLaunchTest(unittest.TestCase):

  def test_kv_and_bare(self):
    got = Q._parse_launch_kwargs(['config=cfg.py', 'force', 'steps=1000'])
    self.assertEqual(got, {'config': 'cfg.py', 'force': True, 'steps': '1000'})

  def test_empty(self):
    self.assertEqual(Q._parse_launch_kwargs(None), {})
    self.assertEqual(Q._parse_launch_kwargs([]), {})

  def test_value_with_equals(self):
    # only the first '=' splits, so a value may contain '='
    self.assertEqual(Q._parse_launch_kwargs(['x=a=b']), {'x': 'a=b'})


class _FlagCtx:
  """Set absl flags for the duration of a with-block, restore after."""

  def __init__(self, **kw):
    self._kw = kw
    self._saved = {}

  def __enter__(self):
    for name, val in self._kw.items():
      flag_name = getattr(Q, name).name          # FlagHolder -> registered name
      self._saved[flag_name] = getattr(flags.FLAGS, flag_name)
      setattr(flags.FLAGS, flag_name, val)
    return self

  def __exit__(self, *a):
    for flag_name, val in self._saved.items():
      setattr(flags.FLAGS, flag_name, val)


class EnqueueDequeueTest(unittest.TestCase):

  def setUp(self):
    self.path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

  def _load(self):
    return RC.load_queue(self.path)

  def test_enqueue_requires_power_and_archs(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER=None, _ARCHS=None):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 2)
    self.assertEqual(self._load(), [])

  def test_enqueue_writes_entry(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7', 'v6p'],
                  _TIER='PROD', _PRIORITY=5, _JOB_NAME='j-a',
                  _LAUNCH=['config=cfg.py', 'force'], _METROS=None,
                  _MAX_PRICE=None, _POWER_TOL=0.5, _TOPOLOGY_LOCKED=False,
                  _WORKDIR='/my/checkout'):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 0)
    q = self._load()
    self.assertEqual(len(q), 1)
    e = q[0]
    # ★job_id/name split (§5.1): --job_name is the readable LABEL (e.name); the
    # job_id is now an opaque minted PK, never the name. This test used to
    # assert job_id == 'j-a' (the old "name IS the id" contract).
    self.assertEqual(e.name, 'j-a')
    self.assertRegex(e.job_id, r'^\d{8}T\d{6}-[0-9a-f]{10}$')
    self.assertNotEqual(e.job_id, e.name)
    self.assertEqual(e.power, 'v7-32')
    self.assertEqual(e.allowed_archs, ['v7', 'v6p'])
    self.assertEqual(e.priority, 5)
    self.assertEqual(e.launch_kwargs, {'config': 'cfg.py', 'force': True})
    self.assertEqual(e.workdir, '/my/checkout')       # explicit workdir kept
    self.assertEqual(e.state, R.JobState.QUEUED)

  def test_enqueue_defaults_workdir_to_cwd(self):
    # monitor v21 fix: enqueuing from the right checkout must capture it, so the
    # router packages the correct source. Unset --workdir defaults to os.getcwd.
    import os
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME='j-cwd', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False, _WORKDIR=None):
      Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(self._load()[0].workdir, os.getcwd())

  def test_enqueue_empty_workdir_opts_into_router_dir(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME='j-empty', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False, _WORKDIR=''):
      Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(self._load()[0].workdir, '')     # explicit '' preserved

  def test_enqueue_rejects_duplicate_job_id(self):
    common = dict(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME='dup', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False)
    with _FlagCtx(**common):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 1)   # duplicate
    self.assertEqual(len(self._load()), 1)

  def test_enqueue_rejects_same_exp_name_and_workdir(self):
    # The 2026-09-17 bug: a re-typed cold-relaunch repeats exp_name + workdir and
    # only job_name may differ. A LIVE row sharing (exp_name, workdir) must be
    # refused even when the job_name differs -- the guard the old name-only check
    # missed on the npu lane, where the persisted name field is empty.
    base = dict(_QUEUE_FILE=self.path, _POWER='h100-8', _ARCHS=['h100'],
                _TIER='PROD', _PRIORITY=0, _METROS=None, _MAX_PRICE=None,
                _POWER_TOL=0.5, _TOPOLOGY_LOCKED=False, _NO_SNAPSHOT=True,
                _WORKDIR='/runs/arm-a', _LAUNCH=['exp_name=arm-a'])
    with _FlagCtx(_JOB_NAME='arm-a', **base):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
    with _FlagCtx(_JOB_NAME='arm-a-again', **base):   # different name, same run
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 1)   # REFUSED
    self.assertEqual(len(self._load()), 1)

  def test_enqueue_force_overrides_live_dup(self):
    # --force (here via '-f' in argv) enqueues a second copy on purpose.
    common = dict(_QUEUE_FILE=self.path, _POWER='h100-8', _ARCHS=['h100'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME='dup', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False, _NO_SNAPSHOT=True,
                  _WORKDIR='/runs/dup')
    with _FlagCtx(**common):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 1)         # refused
      self.assertEqual(Q._cmd_enqueue(['queue_cli', '-f']), 0)   # forced
    self.assertEqual(len(self._load()), 2)

  def test_enqueue_allows_reuse_after_terminal(self):
    # Re-running a FINISHED experiment reuses its name/exp_name freely: a
    # terminal row is history, not a live duplicate.
    common = dict(_QUEUE_FILE=self.path, _POWER='h100-8', _ARCHS=['h100'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME='rerun', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False, _NO_SNAPSHOT=True,
                  _WORKDIR='/runs/rerun')
    with _FlagCtx(**common):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
    q = self._load()
    q[0].state = R.JobState.DONE                 # the first run finished
    RC.save_queue(self.path, q)
    with _FlagCtx(**common):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)   # allowed
    self.assertEqual(len(self._load()), 2)

  def test_enqueue_allows_same_exp_name_different_workdir(self):
    # The pooler case: each commit reuses exp_name but pins a fresh
    # --workdir=runs/<sha>/. Two LIVE rows with the same exp_name and DIFFERENT
    # workdirs are two legitimate runs, so neither is refused.
    def _one(jobname, wd):
      with _FlagCtx(_QUEUE_FILE=self.path, _POWER='h100-8', _ARCHS=['h100'],
                    _TIER='PROD', _PRIORITY=0, _JOB_NAME=jobname,
                    _LAUNCH=['exp_name=poolerrun'], _METROS=None,
                    _MAX_PRICE=None, _POWER_TOL=0.5, _TOPOLOGY_LOCKED=False,
                    _NO_SNAPSHOT=True, _WORKDIR=wd):
        return Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(_one('poolerrun-sha1', '/runs/sha1'), 0)
    self.assertEqual(_one('poolerrun-sha2', '/runs/sha2'), 0)   # allowed
    self.assertEqual(len(self._load()), 2)

  def test_enqueue_requires_job_name(self):
    # ★No auto-mint: an omitted --job_name is a usage error, not a power-hash.
    # This is the inverse of the old test_enqueue_autogenerates_job_id, which
    # asserted the exact behavior we removed -- a job named after nothing.
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v6e-32', _ARCHS=['v6e'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME=None, _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 2)   # REFUSED
    self.assertEqual(self._load(), [])                     # nothing enqueued

  def test_dequeue_removes_by_id(self):
    # seed two
    for jid in ('a', 'b'):
      with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                    _TIER='PROD', _PRIORITY=0, _JOB_NAME=jid, _LAUNCH=None,
                    _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                    _TOPOLOGY_LOCKED=False, _NO_SNAPSHOT=True):
        Q._cmd_enqueue(['queue_cli'])
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_NAME=None):
      # -f: a QUEUED row is in _DEQUEUE_UNSAFE_STATES, so removal needs --force.
      # 'a' is now resolved by resolve_job_refs -- it matches the row whose
      # readable NAME is 'a' (job_id is an opaque minted id you would not type).
      rc = Q._cmd_dequeue(['queue_cli', 'a', '-f'])   # positional name
    self.assertEqual(rc, 0)
    self.assertEqual([e.name for e in self._load()], ['b'])

  def test_dequeue_missing_id_is_error(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_NAME=None):
      rc = Q._cmd_dequeue(['queue_cli', 'nope'])
    self.assertEqual(rc, 1)

  def test_dequeue_no_id_usage_error(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_NAME=None):
      rc = Q._cmd_dequeue(['queue_cli'])
    self.assertEqual(rc, 2)


class MainDispatchTest(unittest.TestCase):

  def test_unknown_subcommand(self):
    self.assertEqual(Q.main(['queue_cli', 'bogus']), 2)

  def test_no_subcommand(self):
    self.assertEqual(Q.main(['queue_cli']), 2)

class EnqueueSnapshotTest(unittest.TestCase):
  """The enqueue-time snapshot: freeze a local copy of the checkout so edits
  between enqueue and build cannot change the code a job ships."""

  def setUp(self):
    import shutil
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
    # An operator-scoped queue file, like ~/.tpu_local_queue.json.
    self.path = os.path.join(self.tmp, '.tpu_local_queue.json')
    # A realistic code checkout to snapshot.
    self.code = os.path.join(self.tmp, 'checkout', 'torch_impl')
    os.makedirs(self.code)
    with open(os.path.join(self.code, 'main.py'), 'w') as f:
      f.write('print(1)\n')
    with open(os.path.join(self.code, 'BUILD'), 'w') as f:
      f.write('py_binary(name="main")\n')

  def _load(self):
    return RC.load_queue(self.path)

  def _common(self, **kw):
    base = dict(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                _TIER='PROD', _PRIORITY=0, _LAUNCH=None, _METROS=None,
                _MAX_PRICE=None, _POWER_TOL=0.5, _TOPOLOGY_LOCKED=False,
                _NO_SNAPSHOT=False)
    base.update(kw)
    return base

  def test_snapshot_root_is_operator_scoped(self):
    # sqa's and npu's queue files must yield DISTINCT snapshot roots, or one
    # operator's snapshots land in the other's dir (same bug the queue file and
    # tmux session each shipped before scoping).
    sqa = Q._snapshot_root('/home/u/.tpu_local_queue.json')
    npu = Q._snapshot_root('/home/u/lyy-work/.npu_local_queue.json')
    self.assertNotEqual(sqa, npu)
    self.assertTrue(sqa.endswith('.tpu_enqueue_snapshots'))
    self.assertTrue(npu.endswith('.npu_enqueue_snapshots'))

  def test_enqueue_creates_snapshot_and_records_it(self):
    with _FlagCtx(**self._common(_JOB_NAME='j-snap', _WORKDIR=self.code)):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 0)
    e = self._load()[0]
    # snapshot_dir is set, distinct from workdir, and on local disk (our tmp).
    self.assertTrue(e.snapshot_dir)
    self.assertNotEqual(e.snapshot_dir, e.workdir)
    self.assertTrue(os.path.isdir(e.snapshot_dir))
    # workdir is still recorded (provenance), and package_dir prefers the snap.
    self.assertEqual(e.workdir, self.code)
    self.assertEqual(R.package_dir(e), e.snapshot_dir)
    # The frozen copy has the real files.
    self.assertTrue(os.path.isfile(os.path.join(e.snapshot_dir, 'main.py')))
    self.assertTrue(os.path.isfile(os.path.join(e.snapshot_dir, 'BUILD')))

  def test_snapshot_is_frozen_against_later_edits(self):
    # THE WHOLE POINT: edit the source AFTER enqueue; the snapshot must not move.
    with _FlagCtx(**self._common(_JOB_NAME='j-frozen', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    snap = self._load()[0].snapshot_dir
    with open(os.path.join(self.code, 'main.py'), 'w') as f:
      f.write('print("EDITED AFTER ENQUEUE")\n')
    frozen = open(os.path.join(snap, 'main.py')).read()
    self.assertEqual(frozen, 'print(1)\n')          # unchanged

  def test_snapshot_excludes_junk(self):
    # The excludes mirror the build-time stage excludes; heavy/irrelevant dirs
    # must not be copied.
    os.makedirs(os.path.join(self.code, '.git'))
    with open(os.path.join(self.code, '.git', 'HEAD'), 'w') as f:
      f.write('ref: x\n')
    with open(os.path.join(self.code, 'model.pt'), 'w') as f:
      f.write('weights')
    os.makedirs(os.path.join(self.code, 'data'))
    with open(os.path.join(self.code, 'data', 'big.rec'), 'w') as f:
      f.write('x' * 1000)
    with _FlagCtx(**self._common(_JOB_NAME='j-excl', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    snap = self._load()[0].snapshot_dir
    self.assertFalse(os.path.exists(os.path.join(snap, '.git')))
    self.assertFalse(os.path.exists(os.path.join(snap, 'model.pt')))
    self.assertFalse(os.path.exists(os.path.join(snap, 'data')))
    self.assertTrue(os.path.isfile(os.path.join(snap, 'main.py')))

  def test_no_snapshot_flag_opts_out(self):
    with _FlagCtx(**self._common(_JOB_NAME='j-live', _WORKDIR=self.code,
                                 _NO_SNAPSHOT=True)):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 0)
    e = self._load()[0]
    self.assertEqual(e.snapshot_dir, '')             # no snapshot
    self.assertEqual(R.package_dir(e), self.code)    # builds the live workdir

  def test_enqueue_refuses_workspace_root(self):
    # A checkout ROOT (has WORKSPACE) is the `tpu enqueue` from ~/work trap:
    # refuse it fail-closed rather than snapshot 400+ top-level dirs.
    root = os.path.join(self.tmp, 'ws')
    os.makedirs(root)
    open(os.path.join(root, 'WORKSPACE'), 'w').close()
    with _FlagCtx(**self._common(_JOB_NAME='j-ws', _WORKDIR=root)):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 2)                           # REFUSED
    self.assertEqual(self._load(), [])               # nothing enqueued

  def test_refused_when_over_size_cap(self):
    # A tree bigger than the cap is refused (fail-closed), naming --no_snapshot.
    orig = Q._SNAPSHOT_MAX_BYTES
    Q._SNAPSHOT_MAX_BYTES = 10        # 10 bytes: our checkout is bigger
    self.addCleanup(lambda: setattr(Q, '_SNAPSHOT_MAX_BYTES', orig))
    with _FlagCtx(**self._common(_JOB_NAME='j-big', _WORKDIR=self.code)):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 2)
    self.assertEqual(self._load(), [])
    # And --no_snapshot lets the same oversized dir through (opt-out honored).
    with _FlagCtx(**self._common(_JOB_NAME='j-big2', _WORKDIR=self.code,
                                 _NO_SNAPSHOT=True)):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)

  def test_dequeue_removes_snapshot(self):
    with _FlagCtx(**self._common(_JOB_NAME='j-del', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    snap = self._load()[0].snapshot_dir
    self.assertTrue(os.path.isdir(snap))
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_NAME=None):
      # -f: a QUEUED row needs --force to dequeue (it is in the unsafe set); the
      # snapshot cleanup runs on the actual removal, which force permits.
      Q._cmd_dequeue(['queue_cli', 'j-del', '-f'])
    self.assertFalse(os.path.exists(snap))           # reclaimed on dequeue

  def test_duplicate_enqueue_does_not_leak_or_clobber_snapshot(self):
    with _FlagCtx(**self._common(_JOB_NAME='dup', _WORKDIR=self.code)):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
      snap = self._load()[0].snapshot_dir
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 1)   # duplicate id
    # the original snapshot survives; the rejected dup left no staging leftover
    self.assertTrue(os.path.isdir(snap))
    root = Q._snapshot_root(self.path)
    leftovers = [n for n in os.listdir(root) if '.staging.' in n]
    self.assertEqual(leftovers, [])

  def test_gc_reclaims_orphan_snapshot(self):
    # A snapshot dir whose job_id is no longer in the queue, older than the
    # grace window, is reclaimed by the next enqueue's GC.
    with _FlagCtx(**self._common(_JOB_NAME='j-keep', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    root = Q._snapshot_root(self.path)
    orphan = os.path.join(root, 'ghost-job')
    os.makedirs(orphan)
    old = time.time() - Q._SNAPSHOT_GC_GRACE_S - 60
    os.utime(orphan, (old, old))
    # A second enqueue triggers the opportunistic GC.
    with _FlagCtx(**self._common(_JOB_NAME='j-keep2', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    self.assertFalse(os.path.exists(orphan))         # orphan reclaimed
    # the two live jobs' snapshots are kept
    live = {e.snapshot_dir for e in self._load()}
    for d in live:
      self.assertTrue(os.path.isdir(d))

  def test_gc_spares_fresh_orphan_within_grace(self):
    # A brand-new dir not yet in the queue may be a concurrent enqueue mid-copy;
    # the grace window must protect it.
    with _FlagCtx(**self._common(_JOB_NAME='j1', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    root = Q._snapshot_root(self.path)
    fresh = os.path.join(root, 'inflight-job')
    os.makedirs(fresh)                                # mtime = now
    with _FlagCtx(**self._common(_JOB_NAME='j2', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    self.assertTrue(os.path.isdir(fresh))            # spared (within grace)

  # --- retention across the post-submit LIVE states (infra bug 2026-09-11) ---
  # The bug: the GC reaped a job's enqueue snapshot the instant the job left the
  # pre-build states (i.e. as soon as it went SUBMITTED). But a SUBMITTED or
  # RUNNING job is still in-queue and can be cancelled / superseded / preempted
  # and re-routed back to QUEUED, and that re-route re-builds from the snapshot.
  # With the snapshot already gone, route_check's workdir guard fires
  # ("package dir does not exist -- re-enqueue from a valid checkout") and parks
  # the job HELD. Fix: keep the snapshot until the job is GENUINELY finished
  # (DONE/FAILED); reap only then. These tests drive the GC predicate directly:
  # the dir is created by a real enqueue, the row's state is set in memory, and
  # the in-memory list is handed to the GC exactly as _cmd_enqueue does.

  def _enqueue_one(self, job_id):
    with _FlagCtx(**self._common(_JOB_NAME=job_id, _WORKDIR=self.code)):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
    # `job_id` here is really the --job_name the caller passed; after the
    # §5.1 split the row's readable label is e.name, while e.job_id is the
    # opaque minted PK. Locate the row by name.
    e = [x for x in self._load() if x.name == job_id][0]
    self.assertTrue(os.path.isdir(e.snapshot_dir))   # created on disk
    return e.snapshot_dir

  def test_reapable_set_excludes_live_states_and_matches_intent(self):
    # Pin the predicate itself: only the two genuinely-finished states are
    # reapable. RUNNING is deliberately NOT reapable even though route_lib puts
    # it in TERMINAL_STATES -- reaping a RUNNING snapshot is the bug.
    self.assertEqual(
        Q._SNAPSHOT_REAPABLE_STATES,
        frozenset({R.JobState.DONE, R.JobState.FAILED}))
    for live in (R.JobState.QUEUED, R.JobState.BUILD_REQUESTED,
                 R.JobState.BUILDING, R.JobState.HELD,
                 R.JobState.BUDGET_DEFERRED, R.JobState.SUBMITTED,
                 R.JobState.RUNNING):
      self.assertNotIn(live, Q._SNAPSHOT_REAPABLE_STATES)
    # The decision hinges on RUNNING being terminal in route_lib but NOT
    # reapable here; assert that gap explicitly so a future 'use TERMINAL_STATES'
    # refactor trips this test rather than silently re-introducing the bug.
    self.assertIn(R.JobState.RUNNING, R.TERMINAL_STATES)
    self.assertNotIn(R.JobState.RUNNING, Q._SNAPSHOT_REAPABLE_STATES)

  def test_gc_retains_submitted_job_snapshot(self):
    # (a) REGRESSION: a SUBMITTED job's snapshot must survive the GC.
    snap = self._enqueue_one('j-sub')
    q = self._load()
    q[0].state = R.JobState.SUBMITTED
    removed = Q._gc_orphan_snapshots(self.path, q, time.time())
    self.assertEqual(removed, [])                     # nothing reaped
    self.assertTrue(os.path.isdir(snap))              # RETAINED
    self.assertEqual(R.package_dir(q[0]), snap)       # workdir guard would pass

  def test_gc_retains_running_job_snapshot(self):
    # (c) REGRESSION: a RUNNING job can still be re-routed (route_check's
    # liveness re-check cancels a nominally-RUNNING wedge and mark_reroute puts
    # it back to QUEUED), so its snapshot must survive too. This is the case the
    # literal 'state not in TERMINAL_STATES' rule would have broken.
    snap = self._enqueue_one('j-run')
    q = self._load()
    q[0].state = R.JobState.RUNNING
    removed = Q._gc_orphan_snapshots(self.path, q, time.time())
    self.assertEqual(removed, [])
    self.assertTrue(os.path.isdir(snap))              # RETAINED

  def test_gc_retains_snapshot_across_submitted_then_requeue(self):
    # (b) THE FULL RE-ROUTE LIFECYCLE: QUEUED -> SUBMITTED -> (re-route) ->
    # QUEUED. The snapshot must survive every leg, so the re-queued job still
    # has a valid package dir to build from.
    snap = self._enqueue_one('j-rr')
    q = self._load()
    q[0].state = R.JobState.SUBMITTED          # handed to XM
    self.assertEqual(Q._gc_orphan_snapshots(self.path, q, time.time()), [])
    self.assertTrue(os.path.isdir(snap))       # survives while SUBMITTED
    q[0].state = R.JobState.QUEUED             # mark_reroute sent it back
    self.assertEqual(Q._gc_orphan_snapshots(self.path, q, time.time()), [])
    self.assertTrue(os.path.isdir(snap))       # still there for the rebuild
    self.assertTrue(os.path.isdir(R.package_dir(q[0])))

  def test_gc_reclaims_terminal_job_snapshot(self):
    # (d) NOT TOO LAX: a job that is genuinely finished (DONE or FAILED) is
    # in-queue but past re-route, so its snapshot IS reaped -- immediately, not
    # after the grace window (the grace window only protects orphans absent from
    # the queue). Both terminal states reap.
    for jid, term in (('j-done', R.JobState.DONE),
                      ('j-failed', R.JobState.FAILED)):
      snap = self._enqueue_one(jid)
      q = self._load()
      # Locate the row by its readable NAME (jid is the --job_name); its opaque
      # job_id is what _gc_orphan_snapshots reports as removed.
      row = [e for e in q if e.name == jid][0]
      row.state = term
      removed = Q._gc_orphan_snapshots(self.path, q, time.time())
      self.assertIn(row.job_id, removed)              # reaped
      self.assertFalse(os.path.exists(snap))          # gone from disk

  def test_negative_control_old_predicate_would_reap_submitted(self):
    # A test that cannot fail proves nothing (engineering.md). This is the
    # negative control for the retention tests above: restore the OLD, buggy
    # reapable set and require the verdict to FLIP -- a SUBMITTED snapshot IS
    # reaped. The old rule kept iff state in {QUEUED, BUILD_REQUESTED, BUILDING,
    # HELD, BUDGET_DEFERRED}; equivalently, reaped iff state in
    # {SUBMITTED, RUNNING, DONE, FAILED}. If this test ever stops reaping, the
    # fault injection has drifted from the real historical bug and the
    # regression tests can no longer be trusted to catch it.
    old_reapable = frozenset({R.JobState.SUBMITTED, R.JobState.RUNNING,
                              R.JobState.DONE, R.JobState.FAILED})
    snap = self._enqueue_one('j-neg')
    q = self._load()
    q[0].state = R.JobState.SUBMITTED
    neg_id = q[0].job_id                             # opaque minted id
    orig = Q._SNAPSHOT_REAPABLE_STATES
    Q._SNAPSHOT_REAPABLE_STATES = old_reapable
    try:
      removed = Q._gc_orphan_snapshots(self.path, q, time.time())
    finally:
      Q._SNAPSHOT_REAPABLE_STATES = orig            # never leak the fault
    self.assertIn(neg_id, removed)                   # the BUG: reaped while live
    self.assertFalse(os.path.exists(snap))
    # And the fix is back in place for the rest of the suite.
    self.assertEqual(Q._SNAPSHOT_REAPABLE_STATES,
                     frozenset({R.JobState.DONE, R.JobState.FAILED}))


class StatusWhyTest(unittest.TestCase):
  """`tpu queue-status` says why a QUEUED row cannot be placed right now,
  instead of one fixed 'no placeable cell' line for every row."""

  def setUp(self):
    fd, self.path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
    lock = self.path + '.lock'
    self.addCleanup(lambda: os.path.exists(lock) and os.remove(lock))
    RC.save_queue(self.path, [R.QueueEntry(job_id='j1', power='v7-32',
                                           allowed_archs=['v7'])])

  def _status(self, oversold):
    avail = {'c7|v7': R.CellAvail(cell='c7', arch='v7', free_chips=320,
                                  oversold=oversold, price=8.0, metro='c7')}

    class _Prov:

      def __init__(self, **unused_kw):
        pass

      def fetch(self):
        return avail, {'v7': 8.0}, {'v7': 320.0}

    buf = io.StringIO()
    with mock.patch.object(Q.avail_provider, 'AvailabilityProvider', _Prov), \
        _FlagCtx(_QUEUE_FILE=self.path), contextlib.redirect_stdout(buf):
      rc = Q._cmd_status(['queue_cli'])
    self.assertEqual(rc, 0)
    return buf.getvalue()

  def test_waiting_row_shows_the_reason(self):
    out = self._status(oversold=True)
    self.assertIn('waiting: v7-32: no free slice', out)
    self.assertNotIn('PLACEABLE now', out)

  def test_negative_control_placeable_row(self):
    out = self._status(oversold=False)
    self.assertIn('PLACEABLE now', out)
    self.assertNotIn('waiting:', out)

  def test_status_uses_the_routers_debited_view(self):
    # c7 shows 10 free v7-32 slices and our own SUBMITTED row already claims
    # one of them in the GQM auction: the status board plans on the router's
    # debited view (one placement view, operator 2026-10-01; in-flight debit,
    # operator 2026-09-30), so it reports 9.
    with _FlagCtx(_FLEET_PAIR_EVENTS_FILE=''):
      self.assertIn('-> c7 (10 free slice(s)', self._status(oversold=False))
      held = R.QueueEntry(job_id='s1', power='v7-32', allowed_archs=['v7'])
      held.state = R.JobState.SUBMITTED
      held.xid = '42'
      held.cell, held.arch, held.chips = 'c7', 'v7', 32
      RC.save_queue(self.path, RC.load_queue(self.path) + [held])
      self.assertIn('-> c7 (9 free slice(s)', self._status(oversold=False))


if __name__ == '__main__':
  unittest.main()
