"""Tests for registry_store: the single write path of the XID registry.

Runs with plain `python3 registry_store_test.py` (stdlib only) and under blaze.
The multi-process tests spawn real processes (this file in --worker mode), so
the locking is exercised across processes, the way the fleet uses it.
"""

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

try:
  from google3.experimental.users.qiaos.tpu_utils import registry_store as rs
except ImportError:  # flat layout
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import registry_store as rs  # type: ignore

FAST = dict(attempts=3, retry_s=0.01)


def _write(path, text):
  with open(path, 'w') as f:
    f.write(text)


def _load(path):
  with open(path) as f:
    return json.load(f)


def _journal(path):
  jp = path + '.journal'
  if not os.path.exists(jp):
    return []
  with open(jp) as f:
    return [json.loads(l) for l in f if l.strip()]


class Base(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.dir = tempfile.mkdtemp(prefix='regstore_test.')
    self.jobs = os.path.join(self.dir, 'jobs.json')
    self.legacy = os.path.join(self.dir, 'jobs_legacy.json')
    self._env = {k: os.environ.pop(k, None)
                 for k in ('TPU_JOBS_FILE', 'TPU_JOBS_LEGACY_FILE')}

  def tearDown(self):
    for k, v in self._env.items():
      if v is not None:
        os.environ[k] = v
    subprocess.run(['rm', '-rf', self.dir], check=False)
    super().tearDown()


class PathTest(Base):

  def test_env_and_default_resolution(self):
    os.environ['TPU_JOBS_FILE'] = self.jobs
    self.assertEqual(rs.jobs_path(), self.jobs)
    self.assertEqual(rs.legacy_path(), self.legacy)       # derived X_legacy.json
    os.environ['TPU_JOBS_LEGACY_FILE'] = '/tmp/elsewhere.json'
    self.assertEqual(rs.legacy_path(), '/tmp/elsewhere.json')
    self.assertEqual(rs.legacy_path(jobs='/a/.npu_jobs.json', legacy=None),
                     '/tmp/elsewhere.json')
    del os.environ['TPU_JOBS_LEGACY_FILE']
    self.assertEqual(rs.legacy_path(jobs='/a/.npu_jobs.json'), '/a/.npu_jobs_legacy.json')


class StrictReadTest(Base):

  def test_missing_file_is_empty(self):
    self.assertEqual(rs.read(self.jobs), {})

  def test_empty_file_is_unreadable_not_empty(self):
    _write(self.jobs, '')
    with self.assertRaises(rs.RegistryUnreadable):
      rs.read(self.jobs, **FAST)

  def test_truncated_json_is_unreadable(self):
    _write(self.jobs, '{"1": {"status": "RUN')
    with self.assertRaises(rs.RegistryUnreadable):
      rs.read(self.jobs, **FAST)

  def test_non_object_is_unreadable(self):
    _write(self.jobs, '[1, 2]')
    with self.assertRaises(rs.RegistryUnreadable):
      rs.read(self.jobs, **FAST)

  def test_reader_waits_out_a_writer_that_truncated(self):
    """The old failure: a reader sees the truncation window and takes it as {}.
    A strict reader retries (releasing the lock) and gets the real content."""
    _write(self.jobs, '')
    writer = threading.Timer(
        0.3, _write, args=(self.jobs, json.dumps({'7': {'status': 'RUNNING'}})))
    writer.start()
    try:
      self.assertEqual(rs.read(self.jobs, attempts=50, retry_s=0.05),
                       {'7': {'status': 'RUNNING'}})
    finally:
      writer.join()


class MutateTest(Base):

  def test_creates_and_journals(self):
    rs.mutate(lambda d: d.__setitem__('1', {'status': 'SUBMITTED'}),
              who='t', path=self.jobs)
    self.assertEqual(_load(self.jobs), {'1': {'status': 'SUBMITTED'}})
    j = _journal(self.jobs)
    self.assertEqual(len(j), 1)
    self.assertEqual(j[0]['who'], 't')
    self.assertEqual(j[0]['added'], ['1'])
    self.assertEqual(_load(self.jobs + '.lastgood'), {'1': {'status': 'SUBMITTED'}})

  def test_no_change_means_no_write(self):
    _write(self.jobs, json.dumps({'1': {'a': 1}}))
    before = os.stat(self.jobs).st_mtime_ns
    time.sleep(0.01)
    rs.mutate(lambda d: None, who='t', path=self.jobs)
    self.assertEqual(os.stat(self.jobs).st_mtime_ns, before)
    self.assertEqual(_journal(self.jobs), [])

  def test_unreadable_registry_is_never_overwritten(self):
    _write(self.jobs, '')
    called = []
    with self.assertRaises(rs.RegistryUnreadable):
      rs.mutate(lambda d: called.append(1) or d.__setitem__('9', {}),
                who='t', path=self.jobs, **FAST)
    self.assertEqual(called, [])                  # fn never saw a fake empty dict
    with open(self.jobs) as f:
      self.assertEqual(f.read(), '')              # file untouched

  def test_create_false_on_missing_is_noop(self):
    self.assertIsNone(rs.mutate(lambda d: d.__setitem__('1', {}), who='t',
                                path=self.jobs, create=False))
    self.assertFalse(os.path.exists(self.jobs))

  def test_shrinking_write_leaves_no_tail(self):
    _write(self.jobs, json.dumps({str(i): {'x': 'y' * 50} for i in range(50)}))
    rs.mutate(lambda d: [d.pop(k) for k in list(d) if k != '3'], who='t', path=self.jobs)
    self.assertEqual(_load(self.jobs), {'3': {'x': 'y' * 50}})

  def test_journal_records_field_changes_and_removals(self):
    _write(self.jobs, json.dumps({'1': {'status': 'SUBMITTED'}, '2': {}}))
    def fn(d):
      d['1']['status'] = 'CANCELLED'
      d.pop('2')
    rs.mutate(fn, who='tpu cancel', path=self.jobs)
    j = _journal(self.jobs)[-1]
    self.assertEqual(j['removed'], ['2'])
    self.assertEqual(j['changed']['1']['status'], ['SUBMITTED', 'CANCELLED'])


class PatchFieldsTest(Base):

  def test_cas_refuses_a_field_changed_since_the_snapshot(self):
    _write(self.jobs, json.dumps({'1': {'status': 'SUBMITTED'}}))
    snap = rs.read(self.jobs)
    # meanwhile a `tpu cancel` lands
    rs.mutate(lambda d: d['1'].__setitem__('status', 'CANCELLED'), who='cancel',
              path=self.jobs)
    out = rs.patch_fields({'1': {'status': (snap['1']['status'], 'RUNNING')}},
                          who='daemon', path=self.jobs)
    self.assertEqual(out['applied'], 0)
    self.assertEqual(len(out['conflicts']), 1)
    self.assertEqual(_load(self.jobs)['1']['status'], 'CANCELLED')

  def test_applies_when_unchanged_and_supports_absent(self):
    _write(self.jobs, json.dumps({'1': {'status': 'SUBMITTED'}}))
    out = rs.patch_fields({'1': {'status': ('SUBMITTED', 'RUNNING'),
                                 'retry_timer_start': (rs.ABSENT, 5)}},
                          who='daemon', path=self.jobs)
    self.assertEqual(out['applied'], 2)
    self.assertEqual(_load(self.jobs)['1'], {'status': 'RUNNING', 'retry_timer_start': 5})

  def test_entry_removed_meanwhile_is_skipped_not_resurrected(self):
    _write(self.jobs, json.dumps({'2': {}}))
    out = rs.patch_fields({'1': {'status': ('SUBMITTED', 'RUNNING')}}, who='d',
                          path=self.jobs)
    self.assertEqual(out['applied'], 0)
    self.assertNotIn('1', _load(self.jobs))


class ArchiveTest(Base):

  def test_moves_to_legacy_and_keeps_siblings(self):
    _write(self.jobs, json.dumps({'1': {'bucket_cp_path': '/cns/x'}, '2': {'a': 1}}))
    _write(self.legacy, json.dumps({'0': {'old': True}}))
    out = rs.archive(['1'], who='t', archived_by='reroute', path=self.jobs,
                     legacy=self.legacy)
    self.assertEqual(out['moved'], ['1'])
    self.assertEqual(_load(self.jobs), {'2': {'a': 1}})
    leg = _load(self.legacy)
    self.assertTrue(leg['0']['old'])
    self.assertEqual(leg['1']['bucket_cp_path'], '/cns/x')
    self.assertEqual(leg['1']['archived_by'], 'reroute')
    self.assertIn('archived_at', leg['1'])

  def test_idempotent(self):
    _write(self.jobs, json.dumps({'1': {}}))
    rs.archive(['1'], who='t', archived_by='x', path=self.jobs, legacy=self.legacy)
    out = rs.archive(['1'], who='t', archived_by='x', path=self.jobs, legacy=self.legacy)
    self.assertEqual(out, {'moved': [], 'merged': []})
    self.assertIn('1', _load(self.legacy))

  def test_extra_and_defaults(self):
    _write(self.jobs, json.dumps({'1': {'bucket_cp_path': '/keep'}}))
    rs.archive(['1'], who='t', archived_by='clear', path=self.jobs,
               legacy=self.legacy,
               extra={'1': {'queue_row': {'job_id': 'j'}}, '5': {'queue_row': {'job_id': 'k'}}},
               defaults={'1': {'bucket_cp_path': '/ignored'}})
    leg = _load(self.legacy)
    self.assertEqual(leg['1']['queue_row'], {'job_id': 'j'})
    self.assertEqual(leg['1']['bucket_cp_path'], '/keep')   # defaults never overwrite
    self.assertEqual(leg['5']['queue_row'], {'job_id': 'k'})  # off-board xid merged

  def test_unreadable_legacy_changes_nothing(self):
    """The old reroute archive read a half-written legacy as {} and wrote back a
    one-entry archive: thousands of records gone. Strict read refuses instead."""
    _write(self.jobs, json.dumps({'1': {'a': 1}}))
    _write(self.legacy, '{"0": {"old"')
    with self.assertRaises(rs.RegistryUnreadable):
      rs.archive(['1'], who='t', archived_by='x', path=self.jobs,
                 legacy=self.legacy)
    self.assertEqual(_load(self.jobs), {'1': {'a': 1}})     # still on the board
    with open(self.legacy) as f:
      self.assertEqual(f.read(), '{"0": {"old"')

  def test_missing_registry_is_noop(self):
    rs.archive(['1'], who='t', archived_by='x', path=self.jobs, legacy=self.legacy)
    self.assertFalse(os.path.exists(self.jobs))
    self.assertFalse(os.path.exists(self.legacy))


class HelpersTest(Base):

  def test_register_fill_keeps_existing_and_overwrite_sets(self):
    _write(self.jobs, json.dumps({'1': {'exp_name': 'keep', 'wandb': {}, 'tier': 'BATCH'}}))
    e = rs.register('1', who='t', path=self.jobs, overwrite={'tier': 'PROD'},
                    fill={'exp_name': 'new', 'wandb': {'id': 'w'}, 'status': 'SUBMITTED'})
    self.assertEqual(e, {'exp_name': 'keep', 'wandb': {'id': 'w'}, 'tier': 'PROD',
                         'status': 'SUBMITTED'})
    self.assertEqual(_load(self.jobs)['1'], e)

  def test_register_creates_missing_registry(self):
    rs.register('7', who='t', path=self.jobs, fill={'status': 'SUBMITTED'})
    self.assertEqual(_load(self.jobs), {'7': {'status': 'SUBMITTED'}})

  def test_mark_cancelled_only_touches_known_xids(self):
    _write(self.jobs, json.dumps({'1': {'status': 'RUNNING'}, '2': {}}))
    self.assertEqual(rs.mark_cancelled(['1', '9'], who='t', path=self.jobs), ['1'])
    d = _load(self.jobs)
    self.assertEqual(d['1']['status'], 'CANCELLED')
    self.assertEqual(d['1']['retry_count'], 5)
    self.assertIn('cancelled_at', d['1'])
    self.assertEqual(d['2'], {})

  def test_cancel_cli(self):
    _write(self.jobs, json.dumps({'1': {'status': 'RUNNING'}}))
    self.assertEqual(rs._cli(['cancel', '--who=tpu cancel', f'--jobs_file={self.jobs}', '1']), 0)
    self.assertEqual(_load(self.jobs)['1']['status'], 'CANCELLED')
    self.assertEqual(_journal(self.jobs)[-1]['who'], 'tpu cancel')

  def test_remove_corpse_never_touches_a_healthy_entry(self):
    _write(self.jobs, json.dumps({'1': {'exp_name': 'x'}, '2': {'status': 'SUBMITTED'}}))
    self.assertTrue(rs.remove_corpse('1', who='t', path=self.jobs))
    self.assertFalse(rs.remove_corpse('2', who='t', path=self.jobs))
    self.assertEqual(_load(self.jobs), {'2': {'status': 'SUBMITTED'}})

  def test_archive_create_writes_valid_registry(self):
    rs.archive([], who='t', archived_by='clear', extra={'5': {'queue_row': {}}},
               path=self.jobs, legacy=self.legacy, create=True)
    self.assertEqual(_load(self.jobs), {})
    self.assertIn('5', _load(self.legacy))

  def test_stash_unwritten(self):
    sp = rs.stash_unwritten({'xid': '1', 'fill': {'a': 1}}, who='t', path=self.jobs)
    with open(sp) as f:
      rec = json.loads(f.readline())
    self.assertEqual((rec['xid'], rec['who']), ('1', 't'))


class RepairTest(Base):

  def test_restores_lastgood_only_when_unreadable(self):
    rs.mutate(lambda d: d.__setitem__('1', {'s': 1}), who='t', path=self.jobs)
    self.assertEqual(rs._cli(['repair', f'--jobs_file={self.jobs}']), 0)
    self.assertEqual(_load(self.jobs), {'1': {'s': 1}})     # readable -> untouched
    _write(self.jobs, '')
    self.assertEqual(rs._cli(['repair', f'--jobs_file={self.jobs}']), 0)
    self.assertEqual(_load(self.jobs), {'1': {'s': 1}})


# --- multi-process ----------------------------------------------------------------
def _worker(mode, path, tag, n):
  """Runs in a child process. Modes:
     new      -- add n unique keys through registry_store.mutate
     compat   -- add n unique keys the way the (correct) older writers do:
                 r+, flock EX on the inode, read, rewrite in place
     reader   -- n strict reads; exit 3 if any read is not a JSON object
     oldtrunc -- the buggy legacy pattern: shared-lock read, release,
                 open('w') (TRUNCATE) before locking, write its snapshot back
  """
  n = int(n)
  if mode in ('new', 'new_tolerant'):
    for i in range(n):
      try:
        rs.mutate(lambda d, i=i: d.__setitem__(f'{tag}-{i}', {'i': i}),
                  who=f'new-{tag}', path=path, attempts=5, retry_s=0.01)
      except rs.RegistryUnreadable:
        if mode == 'new':
          raise
        # new_tolerant: an old truncating writer corrupted the file; refusing
        # to write is the correct outcome, so record it and carry on.
        with open(path + f'.refused.{tag}', 'a') as fh:
          fh.write(f'{i}\n')
  elif mode == 'compat':
    for i in range(n):
      with open(path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.load(f)
        d[f'{tag}-{i}'] = {'i': i}
        f.seek(0)
        f.write(json.dumps(d, indent=2))
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
  elif mode == 'reader':
    for _ in range(n):
      try:
        d = rs.read(path, attempts=5, retry_s=0.01)
      except rs.RegistryUnreadable:
        continue          # refusing is fine; returning a non-dict is not
      if not isinstance(d, dict):
        sys.exit(3)
  elif mode == 'oldtrunc':
    for _ in range(n):
      with open(path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
          d = json.load(f)
        except ValueError:
          d = None
        fcntl.flock(f, fcntl.LOCK_UN)
      if d is None:
        continue
      with open(path, 'w') as f:            # truncates BEFORE the lock
        time.sleep(0.002)
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(d, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)
      time.sleep(0.005)


class MultiProcessTest(Base):

  def _spawn(self, mode, tag, n):
    # Plain `python3 file.py`: re-run this file. Under blaze sys.executable is
    # unset and argv[0] is the test binary itself; both honour --worker.
    cmd = ([sys.executable, os.path.abspath(__file__)] if sys.executable
           else [sys.argv[0]])
    return subprocess.Popen(cmd + ['--worker', mode, self.jobs, tag, str(n)],
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)})

  def test_concurrent_new_and_compat_writers_lose_nothing(self):
    _write(self.jobs, json.dumps({'seed': {}}))
    procs = ([self._spawn('new', f'n{k}', 40) for k in range(4)]
             + [self._spawn('compat', f'c{k}', 40) for k in range(2)]
             + [self._spawn('reader', f'r{k}', 150) for k in range(2)])
    rcs = [p.wait(timeout=180) for p in procs]
    self.assertEqual(rcs, [0] * len(procs))
    d = _load(self.jobs)
    want = {'seed'} | {f'n{k}-{i}' for k in range(4) for i in range(40)} \
        | {f'c{k}-{i}' for k in range(2) for i in range(40)}
    self.assertEqual(set(d), want)

  def test_new_writers_never_mistake_a_truncation_for_empty(self):
    """With a truncate-before-lock writer still running (the pre-migration
    world) the OLD writer does two kinds of damage on its own: it can roll a
    new writer's addition back to its stale snapshot, and -- because it never
    truncates after taking the lock -- it can leave a longer file's tail behind
    its shorter snapshot ('Extra data'). Both go away only when every old
    writer is replaced, which is why the rollout restarts the daemons.

    What a NEW writer must never do in that world is read a truncation window
    or a corrupt file as {} and write back a near-empty registry. It either
    writes a complete registry or refuses. The journal proves it: no
    new-writer record ever removes an entry."""
    _write(self.jobs, json.dumps({f'seed-{i}': {'i': i} for i in range(30)}))
    procs = ([self._spawn('new_tolerant', f'n{k}', 40) for k in range(3)]
             + [self._spawn('oldtrunc', 'o0', 60)]
             + [self._spawn('reader', 'r0', 150)])
    rcs = [p.wait(timeout=180) for p in procs]
    self.assertEqual(rcs, [0] * len(procs))
    new_records = [r for r in _journal(self.jobs) if r['who'].startswith('new-')]
    self.assertTrue(new_records)                  # new writers did write
    for rec in new_records:
      self.assertEqual(rec['removed'], [], rec)
    try:
      d = rs.read(self.jobs, attempts=1)
    except rs.RegistryUnreadable:
      return      # the OLD writer left a torn tail; new writers refused it
    self.assertTrue(all(f'seed-{i}' in d for i in range(30)))

  def test_new_writers_alone_are_lossless_under_heavy_contention(self):
    _write(self.jobs, json.dumps({f'seed-{i}': {'i': i} for i in range(30)}))
    procs = [self._spawn('new', f'n{k}', 60) for k in range(6)] + \
        [self._spawn('reader', f'r{k}', 200) for k in range(2)]
    rcs = [p.wait(timeout=240) for p in procs]
    self.assertEqual(rcs, [0] * len(procs))
    d = _load(self.jobs)
    self.assertEqual(len(d), 30 + 6 * 60)


if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    _worker(*sys.argv[2:6])
    sys.exit(0)
  unittest.main()
