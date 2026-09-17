"""Unit tests for the router tick. Fake provider + fake submitter: no RPC, no
shell, no real queue file except a temp round-trip."""

import os
import tempfile
import json
import subprocess
import unittest

from google3.experimental.users.qiaos.tpu_utils import avail_provider as AP
from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R


# ★TESTS MUST NOT WRITE PRODUCTION STATE. run_reroute() falls back to the real
# `~/.tpu_reroute_history.json` when no history_file is passed, and the global
# brake reads that file: a test run that exercises the cancel path eight times
# fills the hour window and SUSPENDS RE-ROUTING FLEET-WIDE for the next hour.
# Measured 2026-09-05 -- eight rows written by one `python -m unittest` pass,
# exactly REROUTE_GLOBAL_MAX_PER_HOUR, brake engaged, and nothing in the test
# output said so because a green test says nothing about what it touched.
# Redirecting the module constant here fixes every present and FUTURE caller,
# which passing history_file= at each call site does not.
_REAL_REROUTE_HISTORY_FILE: str = RC.REROUTE_HISTORY_FILE
_TMP_REROUTE_HISTORY = None


def setUpModule():
  global _REAL_REROUTE_HISTORY_FILE, _TMP_REROUTE_HISTORY
  _REAL_REROUTE_HISTORY_FILE = RC.REROUTE_HISTORY_FILE
  fh = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
  fh.write(b'[]')
  fh.close()
  _TMP_REROUTE_HISTORY = fh.name
  RC.REROUTE_HISTORY_FILE = fh.name


def tearDownModule():
  RC.REROUTE_HISTORY_FILE = _REAL_REROUTE_HISTORY_FILE
  if _TMP_REROUTE_HISTORY and os.path.exists(_TMP_REROUTE_HISTORY):
    os.unlink(_TMP_REROUTE_HISTORY)


def _entry(job_id='j1', power='v7-32', archs=('v7',), **kw):
  return R.QueueEntry(job_id=job_id, power=power, allowed_archs=list(archs), **kw)


def _avail(cell, arch, free, oversold=False, price=20.0, metro=''):
  return R.CellAvail(cell=cell, arch=arch, free_chips=free, oversold=oversold,
                     price=price, metro=metro or cell)


class _FakeProvider:
  """Stands in for AvailabilityProvider.fetch()."""

  def __init__(self, avail_by_cell, arch_price=None, arch_pool=None):
    self._a = avail_by_cell
    self._p = arch_price or {}
    self._pool = arch_pool or {}

  def fetch(self):
    return self._a, self._p, self._pool


class _FakeSubmitter:
  """Records argv, returns a scripted xid (or None to simulate a dead launch)."""

  def __init__(self, xid='555001', name_lookup=None):
    self.calls = []
    self.cwds = []
    self.job_ids = []
    self.cancels = []
    self._xid = xid
    # What find_xid_by_name should answer. Default: the lookup RAN and saw
    # nothing -- the reading that lets a reclaimed row rebuild. Tests that
    # exercise adoption pass an explicit (xid, how) pair.
    self.name_lookups = []
    self._name_lookup = name_lookup or (None, 'XM lookup ran and found no exact-name match')

  def submit(self, argv, cwd='', on_early_xid=None, job_id=''):
    self.calls.append(argv)
    self.cwds.append(cwd)
    self.job_ids.append(job_id)
    # Mirror production: fire the early-binding callback with the XID the instant
    # the experiment is "created", BEFORE returning the (post-build) result. A
    # scripted None xid means a dead launch -- no experiment, no early callback.
    if on_early_xid is not None and self._xid:
      on_early_xid(self._xid)
    return self._xid, f'Launched experiment {self._xid}' if self._xid else 'no line'

  def cancel(self, xid):
    self.cancels.append(xid)
    return True, 'stopped'

  def find_xid_by_name(self, exp_name, timeout_s=120.0):
    self.name_lookups.append(exp_name)
    return self._name_lookup


class _FakePlacementProbe:
  """Scripted placement_of(): xid -> (cell, arch, chips), or None if unknown.

  Stands in for XManagerPlacementProbe so the adopt/reroute backfill can be
  exercised without an XManager RPC."""

  def __init__(self, by_xid=None):
    self._by_xid = by_xid or {}
    self.calls = []

  def placement_of(self, xid):
    self.calls.append(xid)
    return self._by_xid.get(xid)


class _FakeJobIdProbe:
  """Scripted find_xid_by_jobid(): returns a (xid, placement, status) triple.

  Stands in for XManagerJobIdProbe so the worker's pre-build escape check can be
  exercised without an XManager RPC. `result` is the triple to return; `calls`
  records (job_id, known_xids) so a test can assert the row's known ids were
  passed through (the re-routed-job de-dup)."""

  def __init__(self, result=(None, None, 'NONE')):
    self._result = result
    self.calls = []

  def find_xid_by_jobid(self, job_id, known_xids=None):
    self.calls.append((job_id, list(known_xids or [])))
    return self._result


# (AdoptEscapedBuildTest removed 2026-09-16 with the adopt_check_name /
# adopt_escaped_builds machinery it exercised. The stale-BUILDING resolution it
# used to cover now lives in route_lib.reclaim_stale_building -- reading each
# row's early-bound submission to tell an escaped build from a crashed one --
# and is tested by route_lib_test.ReclaimEarlyBoundTest. The last-resort
# name lookup for a submit that timed out before the early XID line is still
# tested by SubmitTimeoutRecoveryTest below.)

class QueuePersistenceTest(unittest.TestCase):

  def test_round_trip(self):
    path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(os.remove, path)
    entries = [
        _entry('a', power='v7-32', archs=('v7', 'v6p'), priority=3,
               launch_kwargs={'config': 'x.py', 'force': True}),
        _entry('b', power='v6e-32', archs=('v6e',), topology_locked=True),
    ]
    RC.save_queue(path, entries)
    back = RC.load_queue(path)
    self.assertEqual([e.job_id for e in back], ['a', 'b'])
    self.assertEqual(back[0].priority, 3)
    self.assertEqual(back[0].launch_kwargs, {'config': 'x.py', 'force': True})
    self.assertTrue(back[1].topology_locked)
    self.assertEqual(back[0].state, R.JobState.QUEUED)

  def test_missing_file_is_empty(self):
    self.assertEqual(RC.load_queue('/no/such/queue.json'), [])


class ConcurrentWriteTest(unittest.TestCase):
  """The concurrency bug: a route tick that load...RPC...save'd the WHOLE queue
  clobbered rows enqueued during the RPC window. These tests pin the fix
  (merge_and_save_touched + the queue lock) and include a negative control that
  the naive whole-overwrite still loses the row -- so the test can actually fail.
  """

  def _queue_path(self):
    path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(os.remove, path)
    # Clean up the sidecar lockfile too.
    self.addCleanup(lambda: os.path.exists(path + '.lock') and
                    os.remove(path + '.lock'))
    return path

  def test_merge_preserves_concurrently_enqueued_row(self):
    # The exact scenario: a tick snapshots the queue (holding [a]), an enqueue
    # lands [a, b] during the RPC window, then the tick writes back its touched
    # copy of [a]. With merge_and_save_touched, b MUST survive.
    path = self._queue_path()
    RC.save_queue(path, [_entry('a', priority=6)])
    snapshot = RC.load_queue(path)            # tick's in-memory snapshot: [a]
    # ... concurrent enqueue lands during the (simulated) RPC window ...
    with RC.with_queue_lock(path):
      live = RC.load_queue(path)
      live.append(_entry('b', priority=0))
      RC.save_queue(path, live)               # queue now [a, b]
    # ... tick finishes and merge-writes its touched snapshot ([a] mutated) ...
    snapshot[0].last_reason = 'routed this tick'
    RC.merge_and_save_touched(path, snapshot)
    back = {e.job_id: e for e in RC.load_queue(path)}
    self.assertIn('b', back, 'concurrently enqueued row was clobbered')
    self.assertIn('a', back)
    self.assertEqual(back['a'].last_reason, 'routed this tick',
                     'touched row must carry the tick mutation')
    self.assertEqual(back['b'].priority, 0)

  def test_negative_control_whole_overwrite_loses_row(self):
    # Prove the test is real: the OLD pattern (whole-queue save of the stale
    # snapshot) DOES drop the concurrently enqueued row.
    path = self._queue_path()
    RC.save_queue(path, [_entry('a', priority=6)])
    snapshot = RC.load_queue(path)            # [a]
    with RC.with_queue_lock(path):
      live = RC.load_queue(path)
      live.append(_entry('b', priority=0))
      RC.save_queue(path, live)               # [a, b]
    RC.save_queue(path, snapshot)             # OLD BUG: overwrite with stale [a]
    back = {e.job_id: e for e in RC.load_queue(path)}
    self.assertNotIn('b', back,
                     'negative control should reproduce the clobber')

  def test_stale_pass_cannot_undo_worker_submission(self):
    path = self._queue_path()
    building = _entry('a')
    building.state = R.JobState.BUILDING
    RC.save_queue(path, [building, _entry('b')])
    snapshot = RC.load_queue(path)
    baseline = {e.job_id: e.to_dict() for e in snapshot}
    # The pass returns every row, although only b changed. A worker finishes
    # a during its RPC window; the old merge restored BUILDING and lost its XID.
    snapshot[1].last_reason = 'legitimate pass update'
    live = RC.load_queue(path)
    live[0].state = R.JobState.SUBMITTED
    live[0].xid = '287160307'
    RC.save_queue(path, live)
    RC.merge_and_save_touched(path, snapshot, baseline=baseline)
    back = {e.job_id: e for e in RC.load_queue(path)}
    self.assertEqual(back['a'].state, R.JobState.SUBMITTED)
    self.assertEqual(back['a'].xid, '287160307')
    self.assertEqual(back['b'].last_reason, 'legitimate pass update')

  def test_conflicting_mutations_preserve_live_row(self):
    path = self._queue_path()
    RC.save_queue(path, [_entry('a')])
    snapshot = RC.load_queue(path)
    baseline = {e.job_id: e.to_dict() for e in snapshot}
    snapshot[0].state = R.JobState.HELD
    live = RC.load_queue(path)
    live[0].state = R.JobState.SUBMITTED
    live[0].xid = 'worker-xid'
    RC.save_queue(path, live)
    RC.merge_and_save_touched(path, snapshot, baseline=baseline)
    self.assertEqual(RC.load_queue(path)[0].to_dict(), live[0].to_dict())

  def test_snapshot_does_not_resurrect_deleted_row(self):
    path = self._queue_path()
    RC.save_queue(path, [_entry('a')])
    snapshot = RC.load_queue(path)
    baseline = {e.job_id: e.to_dict() for e in snapshot}
    snapshot[0].last_reason = 'stale update'
    RC.save_queue(path, [])
    RC.merge_and_save_touched(path, snapshot, baseline=baseline)
    self.assertEqual(RC.load_queue(path), [])

  def test_snapshot_drop_preserves_concurrently_changed_row(self):
    path = self._queue_path()
    RC.save_queue(path, [_entry('a'), _entry('b')])
    snapshot = RC.load_queue(path)
    baseline = {e.job_id: e.to_dict() for e in snapshot}
    live = RC.load_queue(path)
    live[0].state = R.JobState.SUBMITTED
    live[0].xid = 'worker-xid'
    RC.save_queue(path, live)
    RC.merge_and_save_touched(
        path, snapshot, dropped_job_ids={'a', 'b'}, baseline=baseline)
    self.assertEqual([e.to_dict() for e in RC.load_queue(path)],
                     [live[0].to_dict()])

  def test_merge_drops_removed_ids(self):
    path = self._queue_path()
    RC.save_queue(path, [_entry('a'), _entry('b'), _entry('c')])
    entries = RC.load_queue(path)
    RC.merge_and_save_touched(path, entries, dropped_job_ids={'b'})
    self.assertEqual([e.job_id for e in RC.load_queue(path)], ['a', 'c'])

  def test_concurrent_enqueue_processes_lose_nothing(self):
    # The real acceptance test: fork N processes that each enqueue a distinct id
    # concurrently, while a route-tick-style merge runs in the parent. Every
    # enqueued id must be present at the end -- 0 lost.
    import multiprocessing
    path = self._queue_path()
    RC.save_queue(path, [_entry('seed', priority=9)])

    n = 12

    def _enqueuer(i):
      with RC.with_queue_lock(path):
        entries = RC.load_queue(path)
        entries.append(_entry(f'job{i}', priority=0))
        RC.save_queue(path, entries)

    procs = [multiprocessing.Process(target=_enqueuer, args=(i,))
             for i in range(n)]
    for p in procs:
      p.start()
    # Meanwhile the parent runs several route-tick-style merge-writes on stale
    # snapshots -- exactly the operation that used to clobber enqueues.
    for _ in range(5):
      snap = RC.load_queue(path)
      for e in snap:
        e.last_reason = 'tick'
      RC.merge_and_save_touched(path, snap)
    for p in procs:
      p.join(timeout=30)
    got = {e.job_id for e in RC.load_queue(path)}
    missing = {f'job{i}' for i in range(n)} - got
    self.assertEqual(missing, set(),
                     f'{len(missing)} concurrently enqueued rows were lost')
    self.assertIn('seed', got)


class BuildCmdTest(unittest.TestCase):

  def test_basic_shape(self):
    e = _entry('j1', tier='PROD')
    p = R.Placement(job_id='j1', arch='v7', chips=32, cell='yutulpz',
                    price=20.0, reason='r')
    argv = RC.build_tpu_queue_cmd(p, e, group='9')
    self.assertEqual(argv[:2], ['tpu', 'queue'])
    self.assertIn('--tpu_type=v7-32', argv)
    self.assertIn('--group=9', argv)
    self.assertIn('--cell=yutulpz', argv)
    self.assertIn('--tier=PROD', argv)

  def test_launch_kwargs_forms(self):
    e = _entry('j1', tier='', launch_kwargs={
        'config': 'cfg.py',        # --config=cfg.py
        'force': True,             # --force  (bare)
        'skip_preflight': None,    # --skip_preflight (bare)
        'disabled': False,         # omitted
        '--already_dashed': 'v',   # kept as-is
    })
    p = R.Placement(job_id='j1', arch='v6p', chips=32, cell='nk',
                    price=9.0, reason='r')
    argv = RC.build_tpu_queue_cmd(p, e)
    self.assertIn('--config=cfg.py', argv)
    self.assertIn('--force', argv)
    self.assertIn('--skip_preflight', argv)
    self.assertNotIn('--disabled', argv)
    self.assertNotIn('--disabled=False', argv)
    self.assertIn('--already_dashed=v', argv)
    self.assertNotIn('--tier=', ' '.join(argv))   # empty tier -> no flag


class ExtractXidTest(unittest.TestCase):

  def test_launched_line(self):
    self.assertEqual(RC.extract_xid('foo\nLaunched experiment 12345\nbar'),
                     '12345')

  def test_resume_workunit_line(self):
    self.assertEqual(
        RC.extract_xid('Added 1 work unit(s) to experiment 67890'), '67890')

  def test_ansi_colorized_id(self):
    self.assertEqual(
        RC.extract_xid('Launched experiment \x1b[1m\x1b[34m99999\x1b[0m'),
        '99999')

  def test_no_line(self):
    self.assertIsNone(RC.extract_xid('build failed, SIGBUS'))


class RunTickTest(unittest.TestCase):

  def test_no_queued_jobs(self):
    e = _entry('j1')
    e.state = R.JobState.RUNNING
    prov = _FakeProvider({})
    out, log = RC.run_tick([e], prov, now=0.0)
    self.assertTrue(any('nothing to do' in l for l in log))

  def test_dry_run_does_not_submit(self):
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter()
    out, log = RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=True)
    self.assertEqual(sub.calls, [])                       # nothing submitted
    self.assertEqual(e.state, R.JobState.QUEUED)          # unchanged
    self.assertTrue(any(l.startswith('[DRY]') for l in log))

  def test_live_submit_marks_submitted(self):
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid='777')
    out, log = RC.run_tick([e], prov, now=100.0, submitter=sub, dry_run=False)
    self.assertEqual(len(sub.calls), 1)
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, '777')
    self.assertEqual(e.cell, 'yutulpz')
    self.assertEqual(e.arch, 'v7')
    self.assertEqual(e.chips, 32)
    # submitted_at is "epoch when handed to XM", i.e. AFTER the (blocking)
    # submit, not the epoch the round opened. This fake returns instantly, so
    # the two differ only by the call's own microseconds -- but asserting exact
    # equality here is what encoded the backdating bug as expected behaviour:
    # with a real submitter that blocks for a 900-1700 s build, `now` is stale
    # by exactly that much. See BackdatedSubmittedAtTest for the measured case.
    self.assertAlmostEqual(e.submitted_at, 100.0, delta=1.0)
    self.assertGreaterEqual(e.submitted_at, 100.0)

  def test_workdir_is_passed_to_submitter_as_cwd(self):
    # REGRESSION (monitor v21 field report): the router must package `tpu queue`
    # from the job's OWN checkout, or a run whose config lives in a snapshot dir
    # (not via --config) ships the wrong source. workdir must reach submit(cwd=).
    e = _entry('j1', power='v7-32', archs=('v7',))
    e.workdir = '/some/checkout/dir'
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid='777')
    RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(sub.cwds, ['/some/checkout/dir'])

  def test_no_workdir_passes_empty_cwd(self):
    e = _entry('j1', power='v7-32', archs=('v7',))   # workdir defaults to ''
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid='777')
    RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(sub.cwds, [''])                  # inherit router CWD

  def test_live_submit_no_xid_marks_failed_attempt(self):
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid=None)                        # dead launch
    out, log = RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(e.state, R.JobState.QUEUED)          # stays queued
    self.assertEqual(e.attempts, 1)
    self.assertIsNone(e.xid)

  def test_nothing_placeable_keeps_queued(self):
    e = _entry('j1', power='v7-32', archs=('v7',))
    # only an oversold cell -> no placement
    prov = _FakeProvider(
        {'yulpptr|v7': _avail('yulpptr', 'v7', 320, oversold=True)},
        arch_price={'v7': 20.0}, arch_pool={'v7': 0})
    sub = _FakeSubmitter()
    out, log = RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(sub.calls, [])
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertTrue(any('nothing placeable' in l for l in log))

  def test_topology_locked_freezes_geometry_on_live_submit(self):
    e = _entry('j1', power='v6p-32', archs=('v7', 'v6p'), topology_locked=True)
    prov = _FakeProvider({'c|v7': _avail('c', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid='888')
    RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(e.locked_geometry, '2x4x4')          # frozen from v7-32


class _FakeProbe:
  """Returns a scripted STATUS_* per xid."""

  def __init__(self, by_xid):
    self._by_xid = by_xid

  def status(self, xid):
    return self._by_xid.get(xid, RC.STATUS_UNKNOWN)


class _BudgetRefusedSubmitter:
  """submit() returns no xid but WITH the budget marker (over-bar refusal)."""

  def __init__(self):
    self.calls = []
    self.cwds = []

  def submit(self, argv, cwd='', on_early_xid=None, job_id=''):
    # A budget refusal never creates an experiment, so there is no early XID to
    # bind -- the callback is accepted (production always passes it) but not
    # fired, exactly like a dead launch.
    self.calls.append(argv)
    self.cwds.append(cwd)
    return None, ('[budget check] total projected: 9999 (Limit: 2228)\n'
                  '[[BUDGET_DEFERRED]]\n'
                  '[budget check] ERROR: Budget exceeded for tpu check!')

  def cancel(self, xid):
    """Part of the _Submitter protocol; never exercised by these tests."""
    raise AssertionError(f'cancel({xid}) must not be called in this test')

  def find_xid_by_name(self, exp_name, timeout_s=120.0):
    # A budget refusal never created an experiment, so the lookup RAN and saw
    # nothing -- the reading that lets the row be retried.
    return None, 'XM lookup ran and found no exact-name match'


def _submitted(job_id, xid, cell, submitted_at, **kw):
  e = _entry(job_id, **kw)
  e.state = R.JobState.SUBMITTED
  e.xid = xid
  e.cell = cell
  e.arch = 'v7'
  e.chips = 32
  e.submitted_at = submitted_at
  return e


  def find_xid_by_name(self, exp_name, timeout_s=120.0):
    # A budget refusal never created an experiment, so the lookup RAN and saw
    # nothing -- the reading that lets the row be retried.
    return None, 'XM lookup ran and found no exact-name match'

class ClassifyStateTest(unittest.TestCase):

  def test_pending_only_when_pending(self):
    self.assertEqual(RC.classify_wu_states(True, False, False), RC.STATUS_PENDING)

  def test_running_wins(self):
    self.assertEqual(RC.classify_wu_states(False, True, False), RC.STATUS_RUNNING)

  def test_terminal_wins_over_all(self):
    self.assertEqual(RC.classify_wu_states(True, True, True), RC.STATUS_TERMINAL)

  def test_coming_up_is_running(self):
    # preparing/starting: not pending, not running, not terminal -> RUNNING
    self.assertEqual(RC.classify_wu_states(False, False, False), RC.STATUS_RUNNING)


class RunRerouteTest(unittest.TestCase):

  def test_no_candidate_when_within_deadline(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=100.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=300.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, [])                     # 200s < 600s
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertTrue(any('no SUBMITTED job past' in l for l in log))

  def test_pending_past_deadline_cancels_and_requeues(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, cooldown_s=1800.0, dry_run=False,
                   sleep_fn=lambda _: None)
    self.assertEqual(sub.cancels, ['111'])               # cancelled
    self.assertEqual(e.state, R.JobState.QUEUED)         # back to queue
    self.assertIsNone(e.xid)
    self.assertGreater(e.cooldown_cells.get('yulpptr', 0), 700.0)  # cell cooled

  def test_dry_run_does_not_cancel(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=True,
                            sleep_fn=lambda _: None)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.SUBMITTED)      # untouched
    self.assertTrue(any('[DRY][reroute]' in l for l in log))

  def test_running_now_is_promoted_not_cancelled(self):
    e = _submitted('j1', '111', 'yukulwh', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_unknown_status_never_cancels(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_UNKNOWN})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, [])                     # never cancel blind
    self.assertEqual(e.state, R.JobState.SUBMITTED)

  def test_terminal_marks_failed(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_TERMINAL})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.FAILED)

  def test_cancel_failure_leaves_submitted(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    class _FailCancel(_FakeSubmitter):
      def cancel(self, xid):
        self.cancels.append(xid)
        return False, 'xmanager stop failed'
    sub = _FailCancel()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False, sleep_fn=lambda _: None)
    self.assertEqual(sub.cancels, ['111'])
    self.assertEqual(e.state, R.JobState.SUBMITTED)      # not re-queued on failure

  def test_queued_job_is_not_a_candidate(self):
    e = _entry('j1')                                      # QUEUED, never submitted
    probe = _FakeProbe({})
    _, log = RC.run_reroute([e], now=10000.0, probe=probe,
                            submitter=_FakeSubmitter(), dry_run=False)
    self.assertTrue(any('no SUBMITTED job past' in l for l in log))


class _SeqProbe:
  """Returns a scripted SEQUENCE of STATUS_* per xid, one per call -- so a test
  can make the first probe PENDING and the second RUNNING (the shadow gap)."""

  def __init__(self, seq_by_xid):
    self._seq = {k: list(v) for k, v in seq_by_xid.items()}
    self.calls = {}

  def status(self, xid):
    self.calls[xid] = self.calls.get(xid, 0) + 1
    seq = self._seq.get(xid, [])
    if not seq:
      return RC.STATUS_UNKNOWN
    return seq.pop(0) if len(seq) > 1 else seq[0]  # last value sticks


class _FakeOutputProbe:
  """Returns a scripted latest_mtime per xid (or None = no disk evidence)."""

  def __init__(self, mtime_by_xid):
    self._by_xid = mtime_by_xid

  def latest_mtime(self, entry):
    return self._by_xid.get(entry.xid)


class RerouteHardeningTest(unittest.TestCase):
  """The 2026-08-24 guards: a single PENDING snapshot must not cancel a job that
  is actually alive (BATCH EMA shadow-WU gap -- xid 282605596)."""

  def _no_sleep(self, _):
    pass

  def test_shadow_gap_second_probe_running_is_not_rerouted(self):
    # First probe PENDING (caught in a shadow gap), second probe RUNNING.
    e = _submitted('j1', '111', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'111': [RC.STATUS_PENDING, RC.STATUS_RUNNING]})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=False,
                            output_probe=_FakeOutputProbe({}),  # no disk evidence
                            confirm_gap_s=15.0, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])                    # NOT cancelled
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(probe.calls['111'], 2)              # took the 2nd sample
    self.assertTrue(any('2nd probe' in l for l in log))

  def test_fresh_output_is_not_rerouted_even_if_pending(self):
    # Both probes would say PENDING, but disk shows a write 60s ago -> alive.
    e = _submitted('j1', '222', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'222': [RC.STATUS_PENDING, RC.STATUS_PENDING]})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=False,
                            output_probe=_FakeOutputProbe({'222': 640.0}),  # 60s ago
                            fresh_output_s=1200.0, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])                    # NOT cancelled
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(probe.calls['222'], 1)              # short-circuited before 2nd probe
    self.assertTrue(any('FRESH output' in l for l in log))

  def test_both_pending_no_fresh_output_is_rerouted(self):
    # The genuine stuck case: two PENDING samples, stale output -> DO reroute.
    e = _submitted('j1', '333', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'333': [RC.STATUS_PENDING, RC.STATUS_PENDING]})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, cooldown_s=1800.0, dry_run=False,
                   output_probe=_FakeOutputProbe({'333': None}),  # no output
                   confirm_gap_s=15.0, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, ['333'])               # cancelled (correct)
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertIsNone(e.xid)

  def test_boundary_stale_output_plus_second_pending_still_reroutes(self):
    # v26 edge: output mtime EXACTLY at the freshness boundary (=stale) AND the
    # second probe also PENDING -> must STILL reroute (guards near-miss, not a
    # permanent shield). fresh_output_s=1200, write was exactly 1200s ago.
    e = _submitted('j1', '444', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'444': [RC.STATUS_PENDING, RC.STATUS_PENDING]})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=2000.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, cooldown_s=1800.0, dry_run=False,
                   output_probe=_FakeOutputProbe({'444': 800.0}),  # 1200s ago == boundary
                   fresh_output_s=1200.0, confirm_gap_s=15.0,
                   sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, ['444'])               # boundary=stale -> reroute
    self.assertEqual(e.state, R.JobState.QUEUED)

  def test_terminal_zombie_cleanup_unaffected_by_hardening(self):
    # v26 req 2: TERMINAL->FAILED path must NOT go through the new guards.
    e = _submitted('j1', '555', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'555': [RC.STATUS_TERMINAL]})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False,
                   output_probe=_FakeOutputProbe({'555': 690.0}),  # even fresh output
                   sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.FAILED)         # zombie still cleaned
    self.assertEqual(probe.calls['555'], 1)              # no 2nd probe for TERMINAL


class _FakeRestartProbe:
  """Returns a scripted in-place restart-since-progress count per xid (or None =
  cannot measure)."""

  def __init__(self, count_by_xid):
    self._by_xid = count_by_xid

  def restarts_since_progress(self, entry):
    return self._by_xid.get(entry.xid)


class InplaceRerouteTest(unittest.TestCase):
  """The in-place preemption-thrash path: a nominally-RUNNING row Borg keeps
  restarting on the same slice without progress is pulled back into the auction,
  and the offending cell takes an eviction strike so the next placement avoids
  it. This is the gap the SUBMITTED reroute path and the output-fresh guard
  structurally cannot see (xid 288745238)."""

  def setUp(self):
    super().setUp()
    fh = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    fh.write(b'[]')
    fh.close()
    self._hist = fh.name

  def tearDown(self):
    if os.path.exists(self._hist):
      os.unlink(self._hist)
    super().tearDown()

  def _no_sleep(self, _):
    pass

  def test_thrashing_running_row_is_rerouted_with_eviction_strike(self):
    # RUNNING locally, past the liveness grace, probe still says RUNNING, and the
    # restart probe reports 5 no-progress restarts (>= threshold 3).
    e = _running('j1', '111', 'ej', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute(
        [e], now=4000.0, probe=probe, submitter=sub,
        cooldown_s=1800.0, dry_run=False,
        output_probe=_FakeOutputProbe({'111': 3999.0}),  # 'fresh' -- would promote!
        restart_probe=_FakeRestartProbe({'111': 5}),
        inplace_reroute=True, inplace_restart_threshold=3,
        nominal_running_grace_s=3600.0,
        history_file=self._hist, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, ['111'])               # cancelled
    self.assertEqual(e.state, R.JobState.QUEUED)         # back to the auction
    self.assertIsNone(e.xid)
    self.assertEqual(e.evictions['ej']['strikes'], 1)    # cell blamed
    self.assertGreater(e.cooldown_cells.get('ej', 0), 4000.0)
    self.assertTrue(any('in-place preemption thrash' in l for l in log))
    self.assertTrue(any('eviction strike recorded' in l for l in log))

  def test_below_threshold_is_promoted_not_cancelled(self):
    # Only 2 restarts -> under threshold -> normal promote, no cancel.
    e = _running('j1', '222', 'ej', submitted_at=0.0)
    probe = _FakeProbe({'222': RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    RC.run_reroute(
        [e], now=4000.0, probe=probe, submitter=sub, dry_run=False,
        restart_probe=_FakeRestartProbe({'222': 2}),
        inplace_reroute=True, inplace_restart_threshold=3,
        nominal_running_grace_s=3600.0,
        history_file=self._hist, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])                    # not cancelled
    self.assertEqual(e.state, R.JobState.RUNNING)        # promoted
    self.assertEqual(e.evictions, {})                   # no strike

  def test_none_signal_is_promoted_not_cancelled(self):
    # Probe cannot measure (no attempt logs / lookup failed) -> do nothing.
    e = _running('j1', '333', 'ej', submitted_at=0.0)
    probe = _FakeProbe({'333': RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    RC.run_reroute(
        [e], now=4000.0, probe=probe, submitter=sub, dry_run=False,
        restart_probe=_FakeRestartProbe({'333': None}),
        inplace_reroute=True, inplace_restart_threshold=3,
        nominal_running_grace_s=3600.0,
        history_file=self._hist, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_dry_run_does_not_cancel(self):
    e = _running('j1', '444', 'ej', submitted_at=0.0)
    probe = _FakeProbe({'444': RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute(
        [e], now=4000.0, probe=probe, submitter=sub, dry_run=True,
        restart_probe=_FakeRestartProbe({'444': 9}),
        inplace_reroute=True, inplace_restart_threshold=3,
        nominal_running_grace_s=3600.0,
        history_file=self._hist, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])                    # dry run: no cancel
    self.assertEqual(e.state, R.JobState.RUNNING)        # untouched
    self.assertTrue(any('[DRY][reroute]' in l and 'in-place preemption thrash' in l
                        for l in log))

  def test_disabled_flag_leaves_thrashing_row_alone(self):
    # inplace_reroute=False -> the detector is off; row promoted as before.
    e = _running('j1', '555', 'ej', submitted_at=0.0)
    probe = _FakeProbe({'555': RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    RC.run_reroute(
        [e], now=4000.0, probe=probe, submitter=sub, dry_run=False,
        restart_probe=_FakeRestartProbe({'555': 9}),
        inplace_reroute=False, inplace_restart_threshold=3,
        nominal_running_grace_s=3600.0,
        history_file=self._hist, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_no_restart_probe_is_backward_compatible(self):
    # A caller that passes no restart_probe (old wiring) behaves exactly as
    # before: RUNNING row promoted, nothing new fires.
    e = _running('j1', '666', 'ej', submitted_at=0.0)
    probe = _FakeProbe({'666': RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    RC.run_reroute(
        [e], now=4000.0, probe=probe, submitter=sub, dry_run=False,
        inplace_reroute=True, inplace_restart_threshold=3,
        nominal_running_grace_s=3600.0,
        history_file=self._hist, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_cancel_failure_leaves_row_running(self):
    e = _running('j1', '777', 'ej', submitted_at=0.0)
    probe = _FakeProbe({'777': RC.STATUS_RUNNING})
    class _FailCancel(_FakeSubmitter):
      def cancel(self, xid):
        self.cancels.append(xid)
        return False, 'xmanager stop failed'
    sub = _FailCancel()
    RC.run_reroute(
        [e], now=4000.0, probe=probe, submitter=sub, dry_run=False,
        restart_probe=_FakeRestartProbe({'777': 5}),
        inplace_reroute=True, inplace_restart_threshold=3,
        nominal_running_grace_s=3600.0,
        history_file=self._hist, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, ['777'])
    self.assertEqual(e.state, R.JobState.RUNNING)        # not re-queued on failure


class SubmitterCwdTest(unittest.TestCase):
  """The real Submitter, exercised only on its input-validation path (no shell):
  a non-existent workdir must be refused BEFORE any packaging happens."""

  def test_nonexistent_workdir_refused(self):
    sub = RC.Submitter()
    xid, out = sub.submit(['tpu', 'queue', '--tpu_type=v7-32'],
                          cwd='/no/such/checkout/dir')
    self.assertIsNone(xid)
    self.assertIn('workdir does not exist', out)


class _StaleProbe:
  """Scripted srcfs failure counter for the mode-2 brake test."""

  def __init__(self, counts):
    self._counts = list(counts)
    self._i = 0

  def failure_count(self):
    v = self._counts[min(self._i, len(self._counts) - 1)]
    self._i += 1
    return v


class BuilderSingletonTest(unittest.TestCase):
  """The process-level builder lock: at most one builder per queue file."""

  def setUp(self):
    self.path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
    lk = RC._builder_lockfile(self.path)
    self.addCleanup(lambda: os.path.exists(lk) and os.remove(lk))
    self._held = []
    self.addCleanup(self._release_all)

  def _release_all(self):
    for fh in self._held:
      try:
        fh.close()
      except OSError:
        pass

  def test_second_builder_is_refused_while_first_holds(self):
    # NEGATIVE CONTROL: the whole point of the fix. First builder grabs the
    # lock; a second attempt on the SAME queue file must be refused.
    first = RC._try_lock_builder(self.path, role='dispatch_worker')
    self.assertIsNotNone(first)
    self._held.append(first)
    second = RC._try_lock_builder(self.path, role='worker')
    self.assertIsNone(second)   # refused: someone already builds this queue

  def test_lock_frees_when_holder_releases(self):
    first = RC._try_lock_builder(self.path, role='worker')
    self.assertIsNotNone(first)
    first.close()               # holder exits -> flock released
    second = RC._try_lock_builder(self.path, role='dispatch_worker')
    self.assertIsNotNone(second)  # now claimable
    self._held.append(second)

  def test_lock_is_per_queue_file(self):
    # tpu and npu builders must stay independent: a lock on one queue file must
    # NOT block a builder on a different queue file.
    other = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(other) and os.remove(other))
    olk = RC._builder_lockfile(other)
    self.addCleanup(lambda: os.path.exists(olk) and os.remove(olk))
    a = RC._try_lock_builder(self.path, role='worker')
    b = RC._try_lock_builder(other, role='worker')
    self.assertIsNotNone(a)
    self.assertIsNotNone(b)     # different queue -> independent lock
    self._held += [a, b]

  def test_holder_stamp_records_role_and_pid(self):
    fh = RC._try_lock_builder(self.path, role='dispatch_worker')
    self._held.append(fh)
    holder = RC.read_builder_singleton_holder(self.path)
    self.assertIn('dispatch_worker', holder)
    self.assertIn(f'pid={os.getpid()}', holder)


class SerialWorkerTest(unittest.TestCase):

  def setUp(self):
    self.path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
    lock = self.path + '.lock'
    self.addCleanup(lambda: os.path.exists(lock) and os.remove(lock))

  def _seed(self, entries):
    RC.save_queue(self.path, entries)

  def _load(self):
    return RC.load_queue(self.path)

  def _byid(self, jid):
    return {e.job_id: e for e in self._load()}[jid]

  def _prov(self, cell='yutulpz', arch='v7', free=320):
    return _FakeProvider({f'{cell}|{arch}': _avail(cell, arch, free)},
                         arch_price={arch: 20.0}, arch_pool={arch: free})

  def test_claims_one_and_submits(self):
    self._seed([_entry('a', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid='900')
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w1')
    self.assertEqual(outcome, 'submitted')
    e = self._byid('a')
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, '900')
    self.assertIsNone(e.build_started_at)         # slot released

  def test_records_last_build_duration_on_submit(self):
    # The build-speed viz in `tpu check` reads last_build_duration; it must be
    # written the instant before build_started_at is cleared. claim_for_build
    # stamps build_started_at = now (here 100.0); the fake submit returns
    # instantly, so submitted_now ~= 100.0 and the recorded duration ~= 0.
    # The point of the assertion is that the field is POPULATED (not None) and
    # non-negative -- with a real blocking build it is the true wall-clock time.
    self._seed([_entry('a', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid='900')
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w1')
    self.assertEqual(outcome, 'submitted')
    e = self._byid('a')
    self.assertIsNone(e.build_started_at)              # slot released
    self.assertIsNotNone(e.last_build_duration)        # duration recorded
    self.assertGreaterEqual(e.last_build_duration, 0.0)

  def test_single_build_invariant_blocks_second_claim(self):
    # one already BUILDING (live) + one QUEUED -> worker must NOT start a 2nd
    e_bld = _entry('bld', power='v7-32', archs=('v7',))
    e_bld.state = R.JobState.BUILDING
    e_bld.build_started_at = 99.0
    e_q = _entry('q', power='v7-32', archs=('v7',))
    self._seed([e_bld, e_q])
    sub = _FakeSubmitter(xid='901')
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w2',
        build_stale_s=1800.0)
    self.assertEqual(outcome, 'busy')
    self.assertEqual(sub.calls, [])               # nothing built
    self.assertEqual(self._byid('q').state, R.JobState.QUEUED)  # still queued

  def test_stale_building_is_reclaimed_then_claimed(self):
    e_bld = _entry('old', power='v7-32', archs=('v7',))
    e_bld.state = R.JobState.BUILDING
    e_bld.build_started_at = 0.0                   # ancient
    self._seed([e_bld])
    sub = _FakeSubmitter(xid='902')
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=5000.0, worker_id='w3',
        build_stale_s=1800.0)
    # stale claim reclaimed -> then the same entry (now QUEUED) is built
    self.assertEqual(outcome, 'submitted')
    self.assertEqual(self._byid('old').state, R.JobState.SUBMITTED)

  def test_no_xid_requeues_not_submitted(self):
    self._seed([_entry('z', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid=None)                 # found[]/build crash
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w4')
    self.assertEqual(outcome, 'requeued')
    e = self._byid('z')
    self.assertEqual(e.state, R.JobState.QUEUED)   # NOT submitted
    self.assertEqual(e.attempts, 1)
    self.assertIsNone(e.build_started_at)          # slot released

  def test_nothing_placeable_requeues(self):
    self._seed([_entry('p', power='v7-32', archs=('v7',))])
    prov = _FakeProvider({'x|v7': _avail('x', 'v7', 320, oversold=True)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 0})
    sub = _FakeSubmitter(xid='903')
    outcome, _, _ = RC.run_worker_once(
        self.path, prov, sub, now=100.0, worker_id='w5')
    self.assertEqual(outcome, 'requeued')
    self.assertEqual(sub.calls, [])
    self.assertEqual(self._byid('p').state, R.JobState.QUEUED)

  def test_idle_when_empty(self):
    self._seed([])
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), _FakeSubmitter(), now=0.0, worker_id='w6')
    self.assertEqual(outcome, 'idle')

  def test_srcfs_brake_skips_when_failures_spike(self):
    self._seed([_entry('b', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid='904')
    probe = _StaleProbe([100, 130])   # +30 between polls (>= 20 brake)
    # first poll establishes baseline (100), no brake, builds
    o1, _, fc1 = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w7',
        stage_probe=probe, srcfs_fail_brake=20, last_fail_count=None)
    self.assertEqual(o1, 'submitted')
    # second poll sees +30 -> brake
    self._seed([_entry('b2', power='v7-32', archs=('v7',))])
    o2, log2, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=200.0, worker_id='w7',
        stage_probe=probe, srcfs_fail_brake=20, last_fail_count=fc1)
    self.assertEqual(o2, 'braked')
    self.assertTrue(any('BRAKE' in l for l in log2))

  def test_worker_loop_bounded(self):
    self._seed([_entry('a', power='v7-32', archs=('v7',)),
                _entry('b', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid='905')
    RC.run_worker_loop(
        self.path, provider_factory=lambda: self._prov(),
        submitter=sub, worker_id='wl', poll_s=0.0, max_iterations=2)
    # both drained to SUBMITTED across 2 iterations (serial)
    states = {e.job_id: e.state for e in self._load()}
    self.assertEqual(states['a'], R.JobState.SUBMITTED)
    self.assertEqual(states['b'], R.JobState.SUBMITTED)

  def test_nonexistent_workdir_is_HELD_not_churned(self):
    e = _entry('bad', power='v7-32', archs=('v7',))
    e.workdir = '/no/such/checkout/xyz'
    self._seed([e])
    sub = _FakeSubmitter(xid='906')
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w')
    self.assertEqual(outcome, 'held')
    self.assertEqual(sub.calls, [])                 # never built
    self.assertEqual(self._byid('bad').state, R.JobState.HELD)

  def test_HELD_job_is_not_claimed_again(self):
    # a HELD job must be skipped; a QUEUED sibling is built instead
    e_held = _entry('held', power='v7-32', archs=('v7',))
    e_held.state = R.JobState.HELD
    e_ok = _entry('ok', power='v7-32', archs=('v7',))
    self._seed([e_held, e_ok])
    sub = _FakeSubmitter(xid='907')
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w')
    self.assertEqual(outcome, 'submitted')
    self.assertEqual(self._byid('ok').state, R.JobState.SUBMITTED)
    self.assertEqual(self._byid('held').state, R.JobState.HELD)  # untouched

  def test_max_attempts_moves_to_HELD_not_infinite_requeue(self):
    e = _entry('z', power='v7-32', archs=('v7',))
    self._seed([e])
    sub = _FakeSubmitter(xid=None)                  # every build fails (no XID)
    # attempts: 0->1 (requeue), 1->2 (requeue), 2->3 (>=3 -> HELD)
    for expected in ('requeued', 'requeued', 'held'):
      o, _, _ = RC.run_worker_once(
          self.path, self._prov(), sub, now=100.0, worker_id='w',
          max_build_attempts=3)
      self.assertEqual(o, expected)
    self.assertEqual(self._byid('z').state, R.JobState.HELD)
    self.assertEqual(self._byid('z').attempts, 3)

  def test_requeue_held_via_helper(self):
    e = _entry('h', power='v7-32', archs=('v7',))
    e.state = R.JobState.HELD
    e.attempts = 5
    R.requeue_held(e)
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertEqual(e.attempts, 0)

  # --- Step3 R2: budget refusal is NOT a build failure ---
  def test_budget_deferral_marker_parks_not_attempts(self):
    # submit returns NO xid but WITH the [[BUDGET_DEFERRED]] marker -> the job
    # must land BUDGET_DEFERRED with attempts UNCHANGED (not treated as a
    # build failure). This is the R2 fix.
    e = _entry('bd', power='v7-32', archs=('v7',))
    self._seed([e])
    sub = _BudgetRefusedSubmitter()
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w')
    self.assertEqual(outcome, 'budget_deferred')
    got = self._byid('bd')
    self.assertEqual(got.state, R.JobState.BUDGET_DEFERRED)
    self.assertEqual(got.attempts, 0)              # NOT incremented
    self.assertIsNone(got.build_started_at)         # slot released

  def test_budget_deferral_never_becomes_held(self):
    # Even after many over-budget rounds, a budget-refused job never accrues
    # attempts toward HELD (contrast test_max_attempts_moves_to_HELD).
    e = _entry('bd2', power='v7-32', archs=('v7',))
    self._seed([e])
    sub = _BudgetRefusedSubmitter()
    for _ in range(5):
      # promote back to QUEUED (as the top-of-round would) then re-run
      cur = self._byid('bd2')
      if cur.state == R.JobState.BUDGET_DEFERRED:
        R.promote_deferred([cur]); RC.save_queue(self.path, [cur])
      o, _, _ = RC.run_worker_once(
          self.path, self._prov(), sub, now=100.0, worker_id='w',
          max_build_attempts=3)
      self.assertEqual(o, 'budget_deferred')
    self.assertEqual(self._byid('bd2').attempts, 0)
    self.assertNotEqual(self._byid('bd2').state, R.JobState.HELD)

  def test_real_build_failure_still_attempts(self):
    # a no-XID WITHOUT the marker is still a real failure -> attempts++ (the
    # existing MODE-1 GUARD path is unchanged by the R2 fix).
    e = _entry('rf', power='v7-32', archs=('v7',))
    self._seed([e])
    sub = _FakeSubmitter(xid=None)                  # no marker, no xid
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w')
    self.assertEqual(outcome, 'requeued')
    self.assertEqual(self._byid('rf').attempts, 1)  # real failure counts


class WorkerJobIdRecoveryTest(unittest.TestCase):
  """§5.4 pre-build escape check: before rebuilding a reclaimed row, ask
  XManager (by the row's jobid tag) whether the build already escaped. This is
  the last resort for the ONE window early binding cannot close -- router died
  between create and persist -- and it is fail-closed."""

  def setUp(self):
    self.path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
    lock = self.path + '.lock'
    self.addCleanup(lambda: os.path.exists(lock) and os.remove(lock))

  def _seed(self, entries):
    RC.save_queue(self.path, entries)

  def _byid(self, jid):
    return {e.job_id: e for e in RC.load_queue(self.path)}[jid]

  def _prov(self, cell='yutulpz', arch='v7', free=320):
    return _FakeProvider({f'{cell}|{arch}': _avail(cell, arch, free)},
                         arch_price={arch: 20.0}, arch_pool={arch: free})

  def _reclaimed(self, jid='r', attempts=1):
    # A row reclaim_stale_building left as 'crashed before create': QUEUED, one
    # attempt burned, no live submission.
    e = _entry(jid, power='v7-32', archs=('v7',))
    e.attempts = attempts
    return e

  def test_found_escaped_build_is_adopted_not_rebuilt(self):
    self._seed([self._reclaimed('r')])
    sub = _FakeSubmitter(xid='999')
    probe = _FakeJobIdProbe(('285706173', ('sj', 'v6p', 32), 'FOUND'))
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=5000.0, worker_id='w',
        jobid_probe=probe)
    self.assertEqual(outcome, 'recovered')
    self.assertEqual(sub.calls, [])                    # NEVER rebuilt
    e = self._byid('r')
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, '285706173')
    self.assertEqual((e.cell, e.arch, e.chips), ('sj', 'v6p', 32))

  def test_none_means_nothing_escaped_so_build_proceeds(self):
    self._seed([self._reclaimed('r')])
    sub = _FakeSubmitter(xid='999')
    probe = _FakeJobIdProbe((None, None, 'NONE'))
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=5000.0, worker_id='w',
        jobid_probe=probe)
    self.assertEqual(outcome, 'submitted')             # built normally
    self.assertEqual(self._byid('r').xid, '999')

  def test_ambiguous_holds_for_a_human_never_guesses(self):
    self._seed([self._reclaimed('r')])
    sub = _FakeSubmitter(xid='999')
    probe = _FakeJobIdProbe((None, None, 'AMBIGUOUS'))
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=5000.0, worker_id='w',
        jobid_probe=probe)
    self.assertEqual(outcome, 'held')
    self.assertEqual(sub.calls, [])                    # never rebuilt
    self.assertEqual(self._byid('r').state, R.JobState.HELD)

  def test_unknown_defers_build_never_rebuilds_on_unreadable_lookup(self):
    self._seed([self._reclaimed('r')])
    sub = _FakeSubmitter(xid='999')
    probe = _FakeJobIdProbe((None, None, 'UNKNOWN'))
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=5000.0, worker_id='w',
        jobid_probe=probe)
    self.assertEqual(outcome, 'deferred')
    self.assertEqual(sub.calls, [])                    # never rebuilt
    e = self._byid('r')
    self.assertEqual(e.state, R.JobState.QUEUED)       # slot released
    self.assertEqual(e.attempts, 1)                    # attempts untouched

  def test_fresh_row_skips_the_lookup_entirely(self):
    # attempts == 0: the create/persist-crash state is impossible, so the happy
    # path must pay NO RPC. The probe must not even be called.
    self._seed([self._reclaimed('r', attempts=0)])
    sub = _FakeSubmitter(xid='999')
    probe = _FakeJobIdProbe((None, None, 'NONE'))
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=5000.0, worker_id='w',
        jobid_probe=probe)
    self.assertEqual(outcome, 'submitted')
    self.assertEqual(probe.calls, [])                  # lookup skipped

  def test_row_with_live_xid_skips_the_lookup(self):
    # A row that already holds a live early-bound xid is not in the escape
    # window; the check must be skipped even though attempts > 0.
    e = self._reclaimed('r')
    e.open_creating(xid='111', cell='sj', arch='v7', chips=32, now=4000.0)
    self._seed([e])
    sub = _FakeSubmitter(xid='999')
    probe = _FakeJobIdProbe((None, None, 'NONE'))
    RC.run_worker_once(self.path, self._prov(), sub, now=5000.0,
                       worker_id='w', jobid_probe=probe)
    self.assertEqual(probe.calls, [])                  # lookup skipped

  def test_no_probe_supplied_is_backward_compatible(self):
    # jobid_probe=None (the default) -> the check is inert and the row builds
    # exactly as before this feature.
    self._seed([self._reclaimed('r')])
    sub = _FakeSubmitter(xid='999')
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=5000.0, worker_id='w')
    self.assertEqual(outcome, 'submitted')

  def test_known_xids_are_passed_so_rerouted_history_is_excluded(self):
    # A re-routed row accumulates tagged experiments it already knows about; the
    # worker must hand them to the probe so they are not re-flagged as escaped.
    e = self._reclaimed('r')
    e.open_creating(xid='100', now=3000.0)
    e.submissions[-1].state = 'SUPERSEDED'             # a past, known attempt
    self._seed([e])
    sub = _FakeSubmitter(xid='999')
    probe = _FakeJobIdProbe((None, None, 'NONE'))
    RC.run_worker_once(self.path, self._prov(), sub, now=5000.0,
                       worker_id='w', jobid_probe=probe)
    self.assertEqual(len(probe.calls), 1)
    _jid, known = probe.calls[0]
    self.assertIn('100', known)                        # known id forwarded


class _FakeExp:
  """Minimal stand-in for an xmanager Experiment: an id and launch_args."""

  def __init__(self, experiment_id, launch_args=()):
    self.experiment_id = experiment_id
    self.launch_args = list(launch_args)


class _FakeXmClient:
  """Stand-in for XManagerApi: list_experiments(tags=...) returns scripted exps
  keyed by the tag, or raises to exercise the UNKNOWN path."""

  def __init__(self, by_tag=None, raises=False):
    self._by_tag = by_tag or {}
    self._raises = raises
    self.calls = []

  def list_experiments(self, tags=None):
    self.calls.append(list(tags or []))
    if self._raises:
      raise RuntimeError('XM unreachable')
    out = []
    for t in (tags or []):
      out.extend(self._by_tag.get(t, []))
    return out


class XManagerJobIdProbeTest(unittest.TestCase):
  """The jobid-tag query primitive (§5.4). Injects a fake client so the
  triple-returning, fail-closed, known-id-excluding logic is tested offline."""

  def _probe(self, client):
    p = RC.XManagerJobIdProbe()
    p._client = client            # inject; _get_client returns it as-is
    return p

  def test_found_unique_returns_xid_and_placement(self):
    exp = _FakeExp(285706173,
                   ['--cell=sj', '--tpu_type=v6p-32', '--foo=bar'])
    client = _FakeXmClient({'jobid:j1': [exp]})
    xid, place, status = self._probe(client).find_xid_by_jobid('j1')
    self.assertEqual(status, 'FOUND')
    self.assertEqual(xid, '285706173')
    self.assertEqual(place, ('sj', 'v6p', 32))
    self.assertEqual(client.calls, [['jobid:j1']])   # queried by the tag

  def test_none_when_no_experiment_carries_the_tag(self):
    client = _FakeXmClient({})
    xid, place, status = self._probe(client).find_xid_by_jobid('j1')
    self.assertEqual((xid, place, status), (None, None, 'NONE'))

  def test_known_xids_are_excluded_so_rerouted_history_is_not_reflagged(self):
    # Two tagged experiments: one the row already knows (a past attempt), one it
    # does not. Only the unknown one is 'escaped'.
    known = _FakeExp(100, ['--cell=sj', '--tpu_type=v6p-32'])
    escaped = _FakeExp(285706173, ['--cell=mtv', '--tpu_type=v7-16'])
    client = _FakeXmClient({'jobid:j1': [known, escaped]})
    xid, _, status = self._probe(client).find_xid_by_jobid(
        'j1', known_xids=['100'])
    self.assertEqual(status, 'FOUND')
    self.assertEqual(xid, '285706173')

  def test_all_matches_known_is_none_not_ambiguous(self):
    # A re-routed row whose every tagged experiment is already recorded: nothing
    # escaped, so NONE (build proceeds) -- must NOT read as AMBIGUOUS.
    a = _FakeExp(100)
    b = _FakeExp(200)
    client = _FakeXmClient({'jobid:j1': [a, b]})
    _, _, status = self._probe(client).find_xid_by_jobid(
        'j1', known_xids=['100', '200'])
    self.assertEqual(status, 'NONE')

  def test_two_unknown_matches_is_ambiguous(self):
    a = _FakeExp(100)
    b = _FakeExp(200)
    client = _FakeXmClient({'jobid:j1': [a, b]})
    xid, _, status = self._probe(client).find_xid_by_jobid('j1')
    self.assertEqual(status, 'AMBIGUOUS')
    self.assertIsNone(xid)

  def test_client_exception_is_unknown_not_none(self):
    # 'Could not see it' must be distinguishable from 'not there'.
    client = _FakeXmClient(raises=True)
    self.assertEqual(self._probe(client).find_xid_by_jobid('j1'),
                     (None, None, 'UNKNOWN'))

  def test_empty_job_id_short_circuits_to_none(self):
    client = _FakeXmClient({})
    self.assertEqual(self._probe(client).find_xid_by_jobid(''),
                     (None, None, 'NONE'))
    self.assertEqual(client.calls, [])               # no query made

  def test_found_but_unparseable_launch_args_still_binds_with_no_placement(self):
    exp = _FakeExp(285706173, ['--nonsense'])         # no cell/tpu_type
    client = _FakeXmClient({'jobid:j1': [exp]})
    xid, place, status = self._probe(client).find_xid_by_jobid('j1')
    self.assertEqual(status, 'FOUND')
    self.assertEqual(xid, '285706173')
    self.assertIsNone(place)


class IsBudgetDeferralTest(unittest.TestCase):
  """Step3 R2: the [[BUDGET_DEFERRED]] marker parser."""

  def test_plain_marker(self):
    self.assertTrue(RC.is_budget_deferral('foo\n[[BUDGET_DEFERRED]]\nbar'))

  def test_ansi_wrapped_marker(self):
    self.assertTrue(RC.is_budget_deferral('\x1b[31m[[BUDGET_DEFERRED]]\x1b[0m'))

  def test_absent(self):
    self.assertFalse(RC.is_budget_deferral('Launched experiment 123'))
    self.assertFalse(RC.is_budget_deferral(''))
    self.assertFalse(RC.is_budget_deferral(''))  # None-ish input, typed as str

  def test_substring_not_matched(self):
    # must be its OWN line, not embedded in prose (avoid false positives).
    self.assertFalse(RC.is_budget_deferral('note: [[BUDGET_DEFERRED]] was seen'))


class RunDispatchTest(unittest.TestCase):
  """Step3: run_dispatch_once (router half) -- promote/backpressure/greedy."""

  def setUp(self):
    self.path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
    lock = self.path + '.lock'
    self.addCleanup(lambda: os.path.exists(lock) and os.remove(lock))

  def _seed(self, entries):
    RC.save_queue(self.path, entries)

  def _byid(self, jid):
    return {e.job_id: e for e in RC.load_queue(self.path)}[jid]

  def _budget(self, headroom, per_cost, exempt_types=()):
    """Fake budget_query_fn: fixed headroom; new_cost from per_cost by type."""
    def fn(tpu_type, tier='PROD', lo='', group=''):
      return {'income': 1000.0, 'bar': 100.0, 'current': 100.0 - headroom,
              'headroom': headroom, 'new_cost': per_cost.get(tpu_type, 0.0),
              'exempt': tpu_type in exempt_types, 'fits': True}
    return fn

  def _q(self, jid, power='v7-32', archs=('v7',), priority=0):
    e = _entry(jid, power=power, archs=archs)
    e.state = R.JobState.QUEUED
    e.priority = priority
    return e

  def test_promote_then_dispatch(self):
    d = self._q('d'); d.state = R.JobState.BUDGET_DEFERRED
    self._seed([d])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=self._budget(1000.0, {'v7-32': 10.0}),
        dry_run=False)
    self.assertEqual(out, 'dispatched')
    self.assertEqual(self._byid('d').state, R.JobState.BUILD_REQUESTED)

  def test_backpressure_skips_dispatch(self):
    br = self._q('br'); br.state = R.JobState.BUILD_REQUESTED
    q = self._q('q')
    self._seed([br, q])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=self._budget(1000.0, {}),
        dry_run=False)
    self.assertEqual(out, 'backpressure')
    self.assertEqual(self._byid('q').state, R.JobState.QUEUED)  # not dispatched

  def test_greedy_marks_fit_and_defer(self):
    a = self._q('a', priority=2); b = self._q('b', priority=1)
    self._seed([a, b])
    # headroom 100, each costs 60 -> a fits (60), b deferred (60 > 40 left)
    out, log = RC.run_dispatch_once(
        self.path, now=100.0,
        budget_query_fn=self._budget(100.0, {'v7-32': 60.0}), dry_run=False)
    self.assertEqual(self._byid('a').state, R.JobState.BUILD_REQUESTED)
    self.assertEqual(self._byid('b').state, R.JobState.BUDGET_DEFERRED)

  def test_no_budget_fails_safe(self):
    self._seed([self._q('a')])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=lambda *a, **k: None,
        dry_run=False)
    self.assertEqual(out, 'no-budget')
    self.assertEqual(self._byid('a').state, R.JobState.QUEUED)  # untouched, safe

  def test_dry_run_does_not_mutate(self):
    self._seed([self._q('a')])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=self._budget(1000.0, {'v7-32': 1.0}),
        dry_run=True)
    self.assertEqual(self._byid('a').state, R.JobState.QUEUED)  # unchanged
    self.assertTrue(any('DRY' in l for l in log))

  def test_idle_when_no_queued(self):
    r = self._q('r'); r.state = R.JobState.RUNNING
    self._seed([r])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=self._budget(1000.0, {}),
        dry_run=False)
    self.assertEqual(out, 'idle')


class GroupOrderTest(unittest.TestCase):
  """The place pass tries groups in preference order (e.g. vqfree g5 then g9).

  The loop in _run applies run_tick once per group, each with that group's own
  availability; a job placed by an earlier group is SUBMITTED and the next
  group's tick only sees the QUEUED remainder. These tests pin that composition
  invariant (which is what --group_order relies on) at the run_tick level.
  """

  def _run_group_order(self, entries, providers_by_group, group_order):
    """Mirror _run's place loop: sequential run_tick per group, live submit."""
    submitters = {}
    updated = entries
    for grp in group_order:
      remaining = [e for e in updated if e.state == R.JobState.QUEUED]
      if not remaining:
        break
      sub = _FakeSubmitter(xid=f'xid-{grp}')
      submitters[grp] = sub
      updated, _ = RC.run_tick(
          updated, providers_by_group[grp], now=100.0, submitter=sub,
          dry_run=False, group=grp)
    return updated, submitters

  def test_prefers_first_group_when_it_can_place(self):
    # A job placeable in BOTH g5 and g9 must be taken by g5 (tried first); g9
    # must never see it.
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov5 = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    prov9 = _FakeProvider({'yudfwra|v7': _avail('yudfwra', 'v7', 320)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    _out, subs = self._run_group_order(
        [e], {'5': prov5, '9': prov9}, ['5', '9'])
    self.assertEqual(len(subs['5'].calls), 1, 'g5 should place the job')
    self.assertNotIn('9', subs, 'g9 tick should be skipped: nothing left QUEUED')
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, 'xid-5')

  def test_falls_back_to_second_group_when_first_cannot_place(self):
    # g5 has NO availability; the job must fall through to g9.
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov5 = _FakeProvider({})  # vqfree empty this tick
    prov9 = _FakeProvider({'yudfwra|v7': _avail('yudfwra', 'v7', 320)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    _out, subs = self._run_group_order(
        [e], {'5': prov5, '9': prov9}, ['5', '9'])
    self.assertEqual(subs['5'].calls, [], 'g5 cannot place (no avail)')
    self.assertEqual(len(subs['9'].calls), 1, 'g9 should place the fallback')
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, 'xid-9')

  def test_split_batch_partly_g5_partly_g9(self):
    # Two jobs, g5 can seat only one slice; the other must fall to g9. Neither
    # is lost, neither double-placed.
    e1 = _entry('j1', power='v7-32', archs=('v7',))
    e2 = _entry('j2', power='v7-32', archs=('v7',))
    prov5 = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 32)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 32})
    prov9 = _FakeProvider({'yudfwra|v7': _avail('yudfwra', 'v7', 320)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    out, subs = self._run_group_order(
        [e1, e2], {'5': prov5, '9': prov9}, ['5', '9'])
    placed_g5 = sum(len(subs[g].calls) for g in subs if g == '5')
    placed_g9 = sum(len(subs[g].calls) for g in subs if g == '9')
    self.assertEqual(placed_g5, 1, 'g5 seats exactly one slice')
    self.assertEqual(placed_g9, 1, 'the other falls to g9')
    self.assertTrue(all(e.state == R.JobState.SUBMITTED for e in out))
    self.assertEqual({e.xid for e in out}, {'xid-5', 'xid-9'})


class RunReconcileTest(unittest.TestCase):
  """Step2: XM-truth reconcile pass (run_reconcile) -- R3 zombie cleanup."""

  def _running(self, job_id, xid):
    e = _submitted(job_id, xid, 'yulpptr', submitted_at=0.0)
    e.state = R.JobState.RUNNING
    return e

  def test_zombie_running_marked_failed(self):
    # local RUNNING but XM says terminal -> the 91%-zombie case.
    e = self._running('z1', '111')
    probe = _FakeProbe({'111': RC.STATUS_TERMINAL})
    out, log = RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertTrue(any('zombie' in l for l in log))

  def test_submitted_promoted_to_running(self):
    e = _submitted('p1', '222', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'222': RC.STATUS_RUNNING})
    RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_unknown_never_acts(self):
    # THE safety rule: a probe hiccup must never mark a live job dead.
    e = self._running('u1', '333')
    probe = _FakeProbe({'333': RC.STATUS_UNKNOWN})
    RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_pending_left_for_reroute(self):
    # reconcile leaves genuinely-pending SUBMITTED for the reroute step.
    e = _submitted('q1', '444', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'444': RC.STATUS_PENDING})
    RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.SUBMITTED)

  def test_dry_run_does_not_mutate(self):
    e = self._running('z2', '555')
    probe = _FakeProbe({'555': RC.STATUS_TERMINAL})
    _, log = RC.run_reconcile([e], now=100.0, probe=probe, dry_run=True)
    self.assertEqual(e.state, R.JobState.RUNNING)          # untouched
    self.assertTrue(any('would set' in l for l in log))

  def test_no_xid_skipped(self):
    e = _entry('b1')
    e.state = R.JobState.BUILDING            # BUILDING with no xid
    e.xid = None
    probe = _FakeProbe({})
    RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.BUILDING)         # no XM identity

  def test_terminal_entries_not_touched(self):
    # QUEUED/HELD/DONE/FAILED are not in RECONCILABLE_STATES -> never probed.
    q = _entry('q'); q.state = R.JobState.QUEUED
    h = _entry('h'); h.state = R.JobState.HELD
    probe = _FakeProbe({})
    out, log = RC.run_reconcile([q, h], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(q.state, R.JobState.QUEUED)
    self.assertEqual(h.state, R.JobState.HELD)
    self.assertTrue(any('no non-terminal entries' in l for l in log))

  def test_summary_counts(self):
    zombie = self._running('z', 'z1')
    promo = _submitted('p', 'p1', 'c', submitted_at=0.0)
    unk = self._running('u', 'u1')
    probe = _FakeProbe({'z1': RC.STATUS_TERMINAL, 'p1': RC.STATUS_RUNNING,
                        'u1': RC.STATUS_UNKNOWN})
    _, log = RC.run_reconcile([zombie, promo, unk], now=100.0, probe=probe,
                              dry_run=False)
    summary = [l for l in log if 'checked:' in l]
    self.assertEqual(len(summary), 1)
    self.assertIn('1 zombie->FAILED', summary[0])
    self.assertIn('1 promoted->RUNNING', summary[0])
    self.assertIn('1 UNKNOWN', summary[0])



# --- submit timeout: local failure is not remote absence -------------------
# ★The bug this covers cost real money: `tpu queue` timed out at 1800s, the
# submitter reported "no XID", the worker counted a failed attempt and
# RESUBMITTED -- while the first experiment was already running. Two copies
# billed the same quota, and the orphan had no local row at all.
_PARTIAL_WITH_XID = 'Launched experiment 284946261\nstill building...'


class SubmitTimeoutRecoveryTest(unittest.TestCase):

  def _submitter(self, name_lookup=None):
    s = RC.Submitter()
    if name_lookup is not None:
      s.find_xid_by_name = name_lookup
    return s

  def test_xid_recovered_from_partial_output(self):
    """Cheapest probe: the id was already printed before the timeout."""
    s = self._submitter(lambda n, **kw: (None, 'should not be reached'))
    # Streaming submit hands recovery the partial output as TEXT (it already read
    # the pipe line-by-line); the timeout path is a plain kill, no exception obj.
    xid, out = s._recover_timed_out_xid(
        _PARTIAL_WITH_XID, ['tpu', 'queue', '--exp_name=job_a'])
    self.assertEqual(xid, '284946261')
    self.assertIn('WAS created', out)

  def test_xid_recovered_from_xm_by_name(self):
    """Timeout landed before the id flushed -> ask XManager by name."""
    s = self._submitter(lambda n, **kw: ('284999999', 'XM lookup matched 1'))
    xid, out = s._recover_timed_out_xid(
        '', ['tpu', 'queue', '--exp_name=job_a'])
    self.assertEqual(xid, '284999999')
    self.assertIn('Adopting it', out)

  def test_NC_genuinely_absent_still_reports_no_xid(self):
    """★Negative control: when the job really was not submitted, we must still
    say so -- the fix must not fabricate an XID and strand a QUEUED row."""
    s = self._submitter(lambda n, **kw: (None, 'XM lookup ran and found no exact-name match'))
    xid, out = s._recover_timed_out_xid(
        '', ['tpu', 'queue', '--exp_name=job_a'])
    self.assertIsNone(xid)
    self.assertIn('Treating as not-submitted', out)

  def test_NC_unknown_remote_state_is_named_not_guessed(self):
    """No --exp_name -> we cannot check; say UNKNOWN rather than imply failure."""
    s = self._submitter()
    xid, out = s._recover_timed_out_xid('', ['tpu', 'queue'])
    self.assertIsNone(xid)
    self.assertIn('UNKNOWN', out)

  def test_NC_xm_lookup_failure_does_not_claim_absence(self):
    """If the lookup itself failed, that is not evidence the job is absent."""
    s = self._submitter(lambda n, **kw: (None, 'XM lookup itself timed out; remote state UNKNOWN'))
    xid, out = s._recover_timed_out_xid(
        '', ['tpu', 'queue', '--exp_name=job_a'])
    self.assertIsNone(xid)
    self.assertIn('UNKNOWN', out)


  def test_submit_ITSELF_recovers_on_timeout(self):
    """★End-to-end through submit(), with a REAL timeout -- no monkeypatching.

    NEGATIVE-CONTROL GAP THIS CLOSES: testing `_recover_timed_out_xid` alone
    still passed when the recovery was ripped out of `submit()`, because
    nothing asserted that submit() actually CALLS it. Here the wrapper is a
    throwaway script that prints the XID and then hangs past the deadline --
    exactly the real shape (experiment created early, build still running).
    """
    d = tempfile.mkdtemp()
    wrapper = os.path.join(d, 'fake_wrapper.sh')
    with open(wrapper, 'w') as f:
      f.write('tpu() { echo "Launched experiment 284946261"; sleep 30; }\n')
    self.addCleanup(lambda: os.path.exists(wrapper) and os.remove(wrapper))

    s = RC.Submitter(wrapper_path=wrapper, timeout_s=1.0)
    xid, out = s.submit(['tpu', 'queue', '--exp_name=job_a'])
    self.assertEqual(xid, '284946261',
                     'submit() must adopt the already-created experiment')
    self.assertIn('WAS created', out)

  def test_exp_name_parsing(self):
    self.assertEqual(RC._exp_name_of(['tpu', '--exp_name=abc']), 'abc')
    self.assertIsNone(RC._exp_name_of(['tpu', 'queue']))


class PriorXidsTest(unittest.TestCase):
  """A resubmitted row must keep its earlier XIDs findable."""

  def _entry(self):
    return R.QueueEntry(job_id='j1', power='h100-8', allowed_archs=['h100'])

  def _placement(self):
    return R.Placement(job_id='j1', cell='sh', arch='h100', chips=8,
                       price=4.0, reason='test', geometry=None)

  def test_superseded_xid_is_preserved(self):
    e = self._entry()
    R.apply_placement(e, self._placement(), xid='111', now=0.0)
    self.assertEqual(e.prior_xids, [])
    R.apply_placement(e, self._placement(), xid='222', now=1.0)
    self.assertEqual(e.xid, '222')
    self.assertEqual(e.prior_xids, ['111'], 'the first XID must remain findable')

  def test_resubmitting_same_xid_is_not_recorded_twice(self):
    e = self._entry()
    R.apply_placement(e, self._placement(), xid='111', now=0.0)
    R.apply_placement(e, self._placement(), xid='111', now=1.0)
    self.assertEqual(e.prior_xids, [])

  def test_history_survives_a_round_trip_through_json(self):
    e = self._entry()
    R.apply_placement(e, self._placement(), xid='111', now=0.0)
    R.apply_placement(e, self._placement(), xid='222', now=1.0)
    back = R.QueueEntry.from_dict(json.loads(json.dumps(e.to_dict())))
    self.assertEqual(back.prior_xids, ['111'])

  def test_old_rows_without_the_field_still_load(self):
    """Backward compatibility: a queue written before this field must load."""
    d = self._entry().to_dict()
    d.pop('prior_xids', None)
    back = R.QueueEntry.from_dict(d)
    self.assertEqual(back.prior_xids, [])

class _FakeBorgProbe:
  """Scripted has_running_vmgroup(): True / False / None (could not tell)."""

  def __init__(self, answer):
    self._answer = answer
    self.calls = []

  def has_running_vmgroup(self, entry):
    self.calls.append(entry.job_id)
    return self._answer


def _running(job_id, xid, cell, submitted_at, **kw):
  """A row already promoted to RUNNING -- the shape needs_reroute never sees."""
  e = _entry(job_id, **kw)
  e.state = R.JobState.RUNNING
  e.xid = xid
  e.cell = cell
  e.arch = 'v7'
  e.chips = 32
  e.submitted_at = submitted_at
  return e


class BucketForEntryTest(unittest.TestCase):
  """The disk guard can only see a job whose write location it can resolve."""

  def test_explicit_bucket_wins(self):
    e = _entry('j1')
    e.launch_kwargs = {'bucket': '/cns/is-d/home/qiaos/eqr_data'}
    e.cell = 'lb'  # would resolve to li-d; the explicit flag must outrank it
    self.assertEqual(RC._bucket_for_entry(e), '/cns/is-d/home/qiaos/eqr_data')

  def test_resolved_from_cell_when_no_bucket(self):
    # ★THE REGRESSION THIS FILE EXISTS FOR. A multi-metro job passes no
    # --bucket (a hardcoded one writes cross-metro and gets it pruned), so the
    # old probe read None and the disk guard was dead for exactly the jobs the
    # guides prescribe. `lb` is in metro lpp, whose storage cell is li-d.
    e = _entry('j1')
    e.launch_kwargs = {'exp_name': 'x'}
    e.cell = 'lb'
    self.assertEqual(RC._bucket_for_entry(e),
                     '/cns/li-d/home/qiaos/eqr_data')

  def test_no_cell_no_bucket_is_none(self):
    e = _entry('j1')
    e.launch_kwargs = {}
    e.cell = None
    self.assertIsNone(RC._bucket_for_entry(e))

  def test_unknown_cell_refuses_to_guess(self):
    # An unmeasured cell must NOT fall through to a default prefix: the wrong
    # path is a perfectly valid path, which is how a job wrote across a
    # continent and was deleted mid-run.
    e = _entry('j1')
    e.launch_kwargs = {}
    e.cell = 'no_such_cell_xyz'
    self.assertIsNone(RC._bucket_for_entry(e))


class NominalRunningRerouteTest(unittest.TestCase):
  """XM RUNNING is not 'has chips'. Reproduces xid 286573746: 12 h in one cell,
  every Borg VM group PENDING, zero bytes written, while XManager and the queue
  both said RUNNING and nothing existed that could move it."""

  def setUp(self):
    self._hist = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    self._hist.write(b'[]')
    self._hist.close()
    self.addCleanup(os.unlink, self._hist.name)

  def _run(self, entry, borg_answer, mtime=None, now=7200.0, grace=3600.0,
           dry_run=False, placement_probe=None):
    probe = _FakeProbe({entry.xid: RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    borg = _FakeBorgProbe(borg_answer)
    _, log = RC.run_reroute(
        [entry], now=now, probe=probe, submitter=sub, reroute_after_s=600.0,
        cooldown_s=1800.0, dry_run=dry_run, output_probe=_FakeOutputProbe({entry.xid: mtime}),
        sleep_fn=lambda _: None, history_file=self._hist.name,
        borg_probe=borg, nominal_running_grace_s=grace,
        placement_probe=placement_probe)
    return sub, log, borg

  def test_no_vmgroup_running_and_no_output_is_rerouted(self):
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    sub, log, borg = self._run(e, borg_answer=False, mtime=None)
    self.assertEqual(sub.cancels, ['111'])              # the 12-hour case acts
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertIsNone(e.xid)
    self.assertGreater(e.cooldown_cells.get('sj', 0), 7200.0)
    self.assertTrue(any('nominally RUNNING' in l for l in log))
    self.assertEqual(borg.calls, ['j1'])                # Borg was consulted

  def test_vmgroup_running_is_left_alone(self):
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    sub, log, _ = self._run(e, borg_answer=True, mtime=None)
    self.assertEqual(sub.cancels, [])                   # a live car is untouched
    self.assertEqual(e.state, R.JobState.RUNNING)
    self.assertTrue(any('borg vmgroup RUN' in l for l in log))

  def test_unreadable_borg_fails_open(self):
    # None is 'could not tell', never 'not running'. An unreadable probe must
    # not become a cancellation.
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    sub, log, _ = self._run(e, borg_answer=None, mtime=None)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)
    self.assertTrue(any('borg unreadable' in l for l in log))

  def test_output_on_disk_beats_borg(self):
    # It wrote something, so it had hardware at some point: promote.
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    sub, log, _ = self._run(e, borg_answer=False, mtime=6000.0)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)
    self.assertTrue(any('output written' in l for l in log))

  def test_within_grace_is_left_alone(self):
    # A young row is still coming up; the grace period must gate the verdict.
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    sub, _, borg = self._run(e, borg_answer=False, mtime=None, now=1800.0,
                             grace=3600.0)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)
    self.assertEqual(borg.calls, [])          # not even selected yet

  def test_dry_run_reports_but_does_not_cancel(self):
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    sub, log, _ = self._run(e, borg_answer=False, mtime=None, dry_run=True)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)
    self.assertTrue(any('[DRY][reroute]' in l and 'NO Borg VM group' in l
                        for l in log))

  def test_high_reroute_count_is_still_rechecked_and_rerouted(self):
    # The give-up bound was REMOVED 2026-09-11 (operator request): a high
    # reroute count no longer exempts a nominally-RUNNING row from the liveness
    # recheck. With no Borg VM group and nothing written, it is re-routed like
    # any other dead row -- it is never auto-parked for churning.
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    e.reroutes = 99
    sub, _, borg = self._run(e, borg_answer=False, mtime=None)
    self.assertEqual(sub.cancels, ['111'])
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertEqual(borg.calls, ['j1'])


  def test_cell_less_running_recovers_cell_from_xm_then_verifies(self):
    # ★xid 288485310 EXACTLY. A RUNNING row with no cell (adopted escaped
    # build) is unverifiable until the cell is recovered. With the placement
    # probe supplying the landing, the cell is filled in AND the usual liveness
    # verdict then runs -- here Borg says no VM group is RUN and nothing was
    # written, so it is re-routed like any other wedge.
    e = _running('j1', '111', None, submitted_at=0.0)  # cell=None
    e.arch = None
    e.chips = None
    pp = _FakePlacementProbe({'111': ('sj', 'b200', 4)})
    sub, log, borg = self._run(e, borg_answer=False, mtime=None, placement_probe=pp)
    self.assertEqual(sub.cancels, ['111'])
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertTrue(any('recovered' in l and 'from XM' in l for l in log))
    # Borg was consulted only AFTER the cell was recovered.
    self.assertEqual(borg.calls, ['j1'])

  def test_cell_less_running_unrecoverable_is_treated_as_stuck(self):
    # ★THE FAIL-OPEN FIX. Still cell-less after recovery (XM had nothing, or no
    # probe): the row is STRUCTURALLY unverifiable -- not the transient
    # 'borg unreadable' that the old code promoted forever. Past grace it is
    # cancelled + re-queued instead of wedging.
    e = _running('j1', '111', None, submitted_at=0.0)
    e.arch = None
    e.chips = None
    pp = _FakePlacementProbe({})   # XM cannot supply a placement
    sub, log, _ = self._run(e, borg_answer=None, mtime=None, placement_probe=pp)
    self.assertEqual(sub.cancels, ['111'])
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertTrue(any('NO cell to verify against' in l for l in log))

  def test_cell_less_running_within_grace_is_left_alone(self):
    # The structural-unverifiable verdict still respects the grace period: a
    # young cell-less row is not cancelled on its first observed tick.
    e = _running('j1', '111', None, submitted_at=0.0)
    e.arch = None
    e.chips = None
    sub, _, _ = self._run(e, borg_answer=None, mtime=None, now=1800.0, grace=3600.0)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_cell_recovered_and_borg_running_is_promoted(self):
    # If recovery fills the cell and Borg then confirms a live VM group, the row
    # is promoted (and now carries the cell for every future pass to see).
    e = _running('j1', '111', None, submitted_at=0.0)
    e.arch = None
    e.chips = None
    pp = _FakePlacementProbe({'111': ('sj', 'b200', 8)})
    sub, log, _ = self._run(e, borg_answer=True, mtime=None, placement_probe=pp)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)
    self.assertEqual(e.cell, 'sj')
    self.assertTrue(any('borg vmgroup RUN' in l for l in log))

class NeedsLivenessRecheckTest(unittest.TestCase):

  def test_running_past_grace_selected(self):
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    self.assertTrue(R.needs_liveness_recheck(e, now=4000.0, grace_s=3600.0))

  def test_running_within_grace_not_selected(self):
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    self.assertFalse(R.needs_liveness_recheck(e, now=3000.0, grace_s=3600.0))

  def test_submitted_row_is_not_this_functions_business(self):
    e = _submitted('j1', '111', 'sj', submitted_at=0.0)
    self.assertFalse(R.needs_liveness_recheck(e, now=99999.0, grace_s=3600.0))

  def test_missing_submitted_at_is_not_selected(self):
    e = _running('j1', '111', 'sj', submitted_at=0.0)
    e.submitted_at = None
    self.assertFalse(R.needs_liveness_recheck(e, now=99999.0, grace_s=3600.0))


class ArchiveFinishedQueueRowsTest(unittest.TestCase):
  """`tpu clear`'s queue half: remove FINISHED rows a set of XIDs points at,
  under the queue lock, and report what was archived vs refused."""

  def _write(self, entries):
    path = tempfile.mkstemp(suffix='.json')[1]
    RC.save_queue(path, entries)
    return path

  def _row(self, job_id, state, xid=None, prior=()):
    e = _entry(job_id=job_id)
    e.state = state
    e.xid = xid
    e.prior_xids = list(prior)
    return e

  def test_finished_row_is_removed_from_the_queue(self):
    path = self._write([self._row('j1', R.JobState.DONE, xid='111')])
    res = RC.archive_finished_queue_rows(path, {'111'})
    self.assertEqual([e.job_id for e in res['archived']], ['j1'])
    self.assertEqual(res['refused'], [])
    self.assertEqual(RC.load_queue(path), [])   # actually gone from disk

  def test_live_row_is_left_in_the_queue(self):
    path = self._write([self._row('j1', R.JobState.RUNNING, xid='111')])
    res = RC.archive_finished_queue_rows(path, {'111'})
    self.assertEqual(res['archived'], [])
    self.assertEqual(res['refused'], [('j1', 'RUNNING', '111')])
    self.assertEqual([e.job_id for e in RC.load_queue(path)], ['j1'])  # still there

  def test_only_the_named_finished_rows_go(self):
    path = self._write([
        self._row('done1', R.JobState.DONE, xid='1'),
        self._row('run1', R.JobState.RUNNING, xid='2'),
        self._row('keep', R.JobState.DONE, xid='3'),   # finished but NOT requested
    ])
    res = RC.archive_finished_queue_rows(path, {'1', '2'})
    self.assertEqual([e.job_id for e in res['archived']], ['done1'])
    self.assertEqual(res['refused'], [('run1', 'RUNNING', '2')])
    left = {e.job_id for e in RC.load_queue(path)}
    self.assertEqual(left, {'run1', 'keep'})

  def test_matches_on_prior_xid(self):
    path = self._write(
        [self._row('j1', R.JobState.FAILED, xid='222', prior=('111',))])
    res = RC.archive_finished_queue_rows(path, {'111'})
    self.assertEqual([e.job_id for e in res['archived']], ['j1'])
    self.assertEqual(RC.load_queue(path), [])

  def test_empty_xid_set_is_a_noop(self):
    path = self._write([self._row('j1', R.JobState.DONE, xid='111')])
    res = RC.archive_finished_queue_rows(path, set())
    self.assertEqual(res['archived'], [])
    self.assertEqual([e.job_id for e in RC.load_queue(path)], ['j1'])

  def test_no_match_leaves_queue_intact(self):
    path = self._write([self._row('j1', R.JobState.DONE, xid='111')])
    res = RC.archive_finished_queue_rows(path, {'nope'})
    self.assertEqual(res['archived'], [])
    self.assertEqual(res['matched_xids'], set())
    self.assertEqual([e.job_id for e in RC.load_queue(path)], ['j1'])


class ArchiveReroutedXidTest(unittest.TestCase):
  """_archive_rerouted_xid: move a just-cancelled reroute XID off the registry
  board into the legacy archive. Negative controls come first -- it must NEVER
  raise into the loop, and must touch ONLY the named XID's registry row."""

  def setUp(self):
    super().setUp()
    # Redirect BOTH registry files to temps via env, exactly the vars the helper
    # reads (same as infra_check / the wrapper). Restored in tearDown so no test
    # can leak into ~/.tpu_jobs.json.
    self._saved = {k: os.environ.get(k) for k in
                   ('TPU_JOBS_FILE', 'TPU_JOBS_LEGACY_FILE',
                    'TPU_REROUTE_NO_ARCHIVE')}
    self.jobs = tempfile.mkstemp(suffix='.jobs.json')[1]
    self.legacy = tempfile.mkstemp(suffix='.legacy.json')[1]
    os.environ['TPU_JOBS_FILE'] = self.jobs
    os.environ['TPU_JOBS_LEGACY_FILE'] = self.legacy
    os.environ.pop('TPU_REROUTE_NO_ARCHIVE', None)

  def tearDown(self):
    for k, v in self._saved.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v
    for p in (self.jobs, self.legacy):
      if os.path.exists(p):
        os.unlink(p)
    super().tearDown()

  def _write_jobs(self, d):
    with open(self.jobs, 'w') as f:
      json.dump(d, f)

  def _read_jobs(self):
    with open(self.jobs) as f:
      return json.load(f)

  def _read_legacy(self):
    if not os.path.exists(self.legacy):
      return {}
    with open(self.legacy) as f:
      body = f.read().strip()
    return json.loads(body) if body else {}   # mkstemp leaves a 0-byte file

  def test_archives_named_xid_off_the_board_into_legacy(self):
    self._write_jobs({'111': {'exp_name': 'foo', 'bucket_cp_path': '/cns/x'},
                      '222': {'exp_name': 'bar'}})
    log = []
    RC._archive_rerouted_xid('111', log)
    self.assertNotIn('111', self._read_jobs())            # gone from the board
    self.assertIn('222', self._read_jobs())               # sibling untouched
    leg = self._read_legacy()
    self.assertIn('111', leg)                             # moved, not deleted
    self.assertEqual(leg['111']['bucket_cp_path'], '/cns/x')  # provenance kept
    self.assertEqual(leg['111']['archived_by'], 'reroute')
    self.assertIn('archived_at', leg['111'])
    self.assertTrue(any('archived superseded xid 111' in l for l in log))

  def test_kill_switch_env_disables_archiving(self):
    os.environ['TPU_REROUTE_NO_ARCHIVE'] = '1'
    self._write_jobs({'111': {'exp_name': 'foo'}})
    log = []
    RC._archive_rerouted_xid('111', log)
    self.assertIn('111', self._read_jobs())               # still on the board
    self.assertEqual(self._read_legacy(), {})

  def test_absent_xid_is_idempotent_noop(self):
    self._write_jobs({'222': {'exp_name': 'bar'}})
    log = []
    RC._archive_rerouted_xid('111', log)                  # 111 never on board
    self.assertEqual(self._read_jobs(), {'222': {'exp_name': 'bar'}})
    self.assertEqual(self._read_legacy(), {})

  def test_double_archive_is_idempotent(self):
    self._write_jobs({'111': {'exp_name': 'foo'}})
    RC._archive_rerouted_xid('111', [])
    RC._archive_rerouted_xid('111', [])                   # second call: no-op, no raise
    self.assertNotIn('111', self._read_jobs())
    self.assertIn('111', self._read_legacy())

  def test_missing_registry_file_is_failsafe(self):
    os.unlink(self.jobs)                                  # no registry at all
    log = []
    RC._archive_rerouted_xid('111', log)                 # must not raise
    self.assertFalse(os.path.exists(self.jobs))

  def test_empty_xid_is_a_noop(self):
    self._write_jobs({'111': {'exp_name': 'foo'}})
    RC._archive_rerouted_xid('', [])
    RC._archive_rerouted_xid(None, [])
    self.assertIn('111', self._read_jobs())

  def test_corrupt_registry_is_failsafe(self):
    with open(self.jobs, 'w') as f:
      f.write('{ this is not valid json')
    log = []
    RC._archive_rerouted_xid('111', log)                 # must not raise
    # a half-written registry is left for the daemon, never partially rewritten

  def test_reroute_end_to_end_leaves_no_board_shell(self):
    # The whole point: a real reroute cancel archives the dead xid off the board.
    self._write_jobs({'111': {'exp_name': 'j1', 'status': 'SUBMITTED'}})
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, cooldown_s=1800.0, dry_run=False,
                   sleep_fn=lambda _: None)
    self.assertEqual(sub.cancels, ['111'])               # it did cancel
    self.assertEqual(e.state, R.JobState.QUEUED)         # and re-queue
    self.assertNotIn('111', self._read_jobs())           # AND archive the shell
    self.assertIn('111', self._read_legacy())

  def test_dry_run_reroute_archives_nothing(self):
    self._write_jobs({'111': {'exp_name': 'j1', 'status': 'SUBMITTED'}})
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=True, sleep_fn=lambda _: None)
    self.assertEqual(sub.cancels, [])                     # no cancel in dry-run
    self.assertIn('111', self._read_jobs())              # so no archive either


if __name__ == '__main__':
  unittest.main()
