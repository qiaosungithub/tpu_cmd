#!/usr/bin/env python3
"""Tests for ca_gc.plan / reference resolution. Self-asserting.

The point of GC is to be SAFE, so these tests are mostly about what it must
REFUSE to delete: referenced dirs, recent dirs, non-CA dirs, symlinks, private
temps within grace. Run: python3 ca_gc_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

import ca_gc

_fail = []


def _ck(cond, msg):
  print(f'[{"ok  " if cond else "FAIL"}] {msg}')
  if not cond:
    _fail.append(msg)


def _mkdir(parent, name, age_days=None):
  p = os.path.join(parent, name)
  os.makedirs(p, exist_ok=True)
  with open(os.path.join(p, 'BUILD'), 'w') as f:
    f.write('x\n')
  if age_days is not None:
    t = time.time() - age_days * 86400
    os.utime(p, (t, t))
  return p


def main():
  tmp = tempfile.mkdtemp(prefix='cagc_test_')
  try:
    parent = os.path.join(tmp, 'eqr_jax_final_stages')
    os.makedirs(parent)

    # old unreferenced CA dir -> deletable
    _mkdir(parent, 'eqr_run_ca_aaaaaaaaaaaa', age_days=30)
    # recent unreferenced CA dir -> keep (too new)
    _mkdir(parent, 'eqr_run_ca_bbbbbbbbbbbb', age_days=1)
    # old but REFERENCED CA dir -> keep
    _mkdir(parent, 'eqr_run_ca_cccccccccccc', age_days=30)
    # a legacy timestamped stagedir (not CA-named) -> keep (not our business)
    _mkdir(parent, 'eqr_run_260901_120000_abcdef', age_days=30)
    # a stuck private temp, old -> deletable (leftover)
    _mkdir(parent, 'eqr_run_ca_dddddddddddd.tmp.123.deadbe', age_days=2)
    # a private temp within grace -> keep (a build may be mid-stage)
    _mkdir(parent, 'eqr_run_ca_eeeeeeeeeeee.tmp.456.beefca', age_days=0)
    # an old fallback dir, unreferenced -> deletable
    _mkdir(parent, 'eqr_run_ca_ffffffffffff_fb_260901_1abc', age_days=5)
    # a random non-CA dir -> keep
    _mkdir(parent, 'some_other_dir', age_days=99)
    # a symlink to an old CA-looking name -> keep (never follow/delete symlinks)
    os.symlink(os.path.join(parent, 'eqr_run_ca_aaaaaaaaaaaa'),
               os.path.join(parent, 'eqr_run_ca_999999999999'))

    referenced = {'eqr_run_ca_cccccccccccc'}
    to_delete, to_keep = ca_gc.plan(parent, referenced, min_age_days=7.0,
                                    tmp_grace_hours=12.0)
    dset = {n for n, _ in to_delete}
    kset = {n for n, _ in to_keep}

    _ck('eqr_run_ca_aaaaaaaaaaaa' in dset, 'old unreferenced CA dir -> delete')
    _ck('eqr_run_ca_bbbbbbbbbbbb' in kset, 'recent CA dir -> keep')
    _ck('eqr_run_ca_cccccccccccc' in kset, 'referenced CA dir -> keep (even if old)')
    _ck('eqr_run_260901_120000_abcdef' in kset, 'legacy timestamp dir -> keep (not CA)')
    _ck('eqr_run_ca_dddddddddddd.tmp.123.deadbe' in dset, 'old stuck temp -> delete')
    _ck('eqr_run_ca_eeeeeeeeeeee.tmp.456.beefca' in kset, 'temp within grace -> keep')
    _ck('eqr_run_ca_ffffffffffff_fb_260901_1abc' in dset, 'old fallback -> delete')
    _ck('some_other_dir' in kset, 'non-CA dir -> keep')
    _ck('eqr_run_ca_999999999999' in kset, 'symlink -> keep (never delete a symlink)')

    # reference resolution reads queue + registry by basename
    qf = os.path.join(tmp, 'q.json')
    jf = os.path.join(tmp, 'jobs.json')
    lf = os.path.join(tmp, 'legacy.json')
    with open(qf, 'w') as f:
      json.dump([{'stagedir': 'experimental/qiaos/eqr_jax_final_stages/eqr_run_ca_aaaaaaaaaaaa',
                  'state': 'QUEUED'}], f)
    with open(jf, 'w') as f:
      json.dump({'123': {'stagedir': 'x/y/eqr_run_ca_cccccccccccc', 'status': 'RUNNING'}}, f)
    with open(lf, 'w') as f:
      json.dump({}, f)
    ref, errs = ca_gc._load_referenced_stagedirs(qf, jf, lf)
    _ck(not errs, 'clean queue+registry -> no errors')
    _ck('eqr_run_ca_aaaaaaaaaaaa' in ref, 'queue stagedir resolved by basename')
    _ck('eqr_run_ca_cccccccccccc' in ref, 'registry stagedir resolved by basename')

    # now the previously-deletable aaa- dir is referenced -> must be kept
    to_delete2, _ = ca_gc.plan(parent, ref, min_age_days=7.0, tmp_grace_hours=12.0)
    _ck('eqr_run_ca_aaaaaaaaaaaa' not in {n for n, _ in to_delete2},
        'a dir referenced by the queue is no longer a delete candidate')

    # an unreadable registry -> errors (caller must refuse to delete)
    bad = os.path.join(tmp, 'bad.json')
    with open(bad, 'w') as f:
      f.write('{ this is not json')
    _, errs2 = ca_gc._load_referenced_stagedirs(qf, bad, lf)
    _ck(bool(errs2), 'unreadable registry -> errors reported (=> fail-safe refuse)')

    # missing files are fine (not an error)
    _, errs3 = ca_gc._load_referenced_stagedirs(
        os.path.join(tmp, 'nope.json'), os.path.join(tmp, 'nope2.json'),
        os.path.join(tmp, 'nope3.json'))
    _ck(not errs3, 'missing queue/registry files are not errors (nothing referenced)')

    # final safety guard rejects a non-CA path and anything outside parent
    _ck(ca_gc._safe_to_remove(os.path.join(parent, 'eqr_run_ca_aaaaaaaaaaaa'), parent),
        'safe_to_remove accepts a real CA dir under parent')
    _ck(not ca_gc._safe_to_remove(os.path.join(parent, 'some_other_dir'), parent),
        'safe_to_remove rejects a non-CA name')
    _ck(not ca_gc._safe_to_remove(tmp, parent),
        'safe_to_remove rejects a path outside parent')

  finally:
    shutil.rmtree(tmp, ignore_errors=True)

  print()
  if _fail:
    print(f'FAILED: {len(_fail)} check(s):')
    for m in _fail:
      print('  -', m)
    return 1
  print('ALL CHECKS PASSED')
  return 0


if __name__ == '__main__':
  import sys
  sys.exit(main())
