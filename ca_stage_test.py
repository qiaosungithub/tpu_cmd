#!/usr/bin/env python3
"""Fault-injection tests for ca_stage.publish (the atomic-publish protocol).

Self-asserting: exits nonzero on the first failure. Run:
    python3 ca_stage_test.py

These tests target the operator's hard constraint directly: a staging dir must
survive perfectly, never be polluted, never be overwritten, and correctness must
NOT depend on a lock. Each test names the invariant it defends (I1..I5 from
ca_stage.py).

Uses real rsync on tiny trees (faithful to production) plus monkeypatched seams
to inject the failures that are otherwise impossible to reproduce on demand:
truncated stage, a lost publish race, an invalid pre-existing final.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

import ca_stage
import stage_content_hash as sch

_fail = []


def _ck(cond, msg):
  print(f'[{"ok  " if cond else "FAIL"}] {msg}')
  if not cond:
    _fail.append(msg)


def _mk_source(root, *, label_placeholder=True, extra=None, yml='lr: 0.1\n'):
  """A minimal but realistic stage source: BUILD, entry src, config.sh, a yml."""
  os.makedirs(os.path.join(root, 'configs'), exist_ok=True)
  with open(os.path.join(root, 'BUILD'), 'w') as f:
    f.write('pytype_binary(\n    name = "main",\n    srcs = ["main_eqr.py"],\n'
            '    data = glob(["configs/**/*.yml"]),\n)\n')
  with open(os.path.join(root, 'main_eqr.py'), 'w') as f:
    f.write('print("hi")\n')
  label = 'export TARGET_LABEL="//PLACEHOLDER:main"\n' if label_placeholder else ''
  with open(os.path.join(root, 'config.sh'), 'w') as f:
    f.write('export PROJECT_NAME="elt-dit"\n' + label)
  with open(os.path.join(root, 'configs', 'base.yml'), 'w') as f:
    f.write(yml)
  if extra:
    for rel, content in extra.items():
      p = os.path.join(root, rel)
      os.makedirs(os.path.dirname(p), exist_ok=True)
      with open(p, 'w') as f:
        f.write(content)


REL_PREFIX = 'experimental/qiaos/eqr_jax_final_stages'


def main():
  tmp = tempfile.mkdtemp(prefix='ca_test_')
  try:
    src = os.path.join(tmp, 'src')
    parent = os.path.join(tmp, 'parent')
    os.makedirs(src)
    os.makedirs(parent)
    _mk_source(src)

    # --- 1. basic publish: new content -> creates final, not reused/fallback ---
    r1 = ca_stage.publish(src, parent, REL_PREFIX)
    _ck(os.path.isdir(r1.dir), 'publish creates the final dir')
    _ck(not r1.reused and not r1.fallback, 'first publish: reused=0 fallback=0')
    _ck(os.path.basename(r1.dir) == f'eqr_run_ca_{r1.sha[:12]}',
        'final dir is eqr_run_ca_<sha12> (keeps eqr_run_ guards working)')
    ok, why = ca_stage.verify_dir(r1.dir, r1.rel, r1.sha)
    _ck(ok, f'published dir verifies ({why})')
    # TARGET_LABEL rewritten to the FINAL label (build correctness).
    with open(os.path.join(r1.dir, 'config.sh')) as f:
      _ck(f'//{r1.rel}:main' in f.read(), 'TARGET_LABEL rewritten to final label')
    # marker == full sha
    with open(os.path.join(r1.dir, ca_stage._MARKER)) as f:
      _ck(f.read().strip() == r1.sha, 'marker equals full content sha (I4/I5)')

    # --- 2. reuse: identical content, second call -> reused, no new dir -------
    before = sorted(os.listdir(parent))
    mtime_before = os.path.getmtime(r1.dir)
    marker_ino = os.stat(os.path.join(r1.dir, ca_stage._MARKER)).st_ino
    time.sleep(0.02)
    r2 = ca_stage.publish(src, parent, REL_PREFIX)
    _ck(r2.reused and r2.dir == r1.dir, 'identical content reuses the same dir')
    _ck(sorted(os.listdir(parent)) == before, 'reuse creates NO new dir/temp')
    _ck(os.path.getmtime(r1.dir) == mtime_before,
        'reuse does NOT mutate the published dir mtime (I2: write-once)')
    _ck(os.stat(os.path.join(r1.dir, ca_stage._MARKER)).st_ino == marker_ino,
        'reuse does NOT rewrite the marker (published dir untouched)')

    # --- 3. different content -> different dir --------------------------------
    src2 = os.path.join(tmp, 'src2')
    os.makedirs(src2)
    _mk_source(src2, yml='lr: 0.2\n')  # one byte differs
    r3 = ca_stage.publish(src2, parent, REL_PREFIX)
    _ck(r3.dir != r1.dir and r3.sha != r1.sha,
        'different content -> different sha -> different dir (no collision)')
    _ck(os.path.isdir(r1.dir) and os.path.isdir(r3.dir),
        'both distinct dirs coexist (no overwrite)')

    # --- 4. lost race, IDENTICAL content: someone publishes final first -------
    # Remove r1's dir, then inject a hook that recreates a VALID final right
    # before our rename, simulating a concurrent builder winning the publish.
    src3 = os.path.join(tmp, 'src3')
    os.makedirs(src3)
    _mk_source(src3, yml='race: 1\n')
    full3 = sch.hash_tree(src3)
    fname3 = ca_stage.final_dirname(full3)
    final3 = os.path.join(parent, fname3)
    rel3 = f'{REL_PREFIX}/{fname3}'

    def _win_the_race(final_path):
      # Build a VALID final out-of-band (as a concurrent winner would).
      if os.path.exists(final_path):
        return
      tmpwin = final_path + '.win'
      shutil.copytree(src3, tmpwin)
      with open(os.path.join(tmpwin, 'config.sh'), 'w') as f:
        f.write('export PROJECT_NAME="elt-dit"\n'
                f'export TARGET_LABEL="//{rel3}:main"\n')
      with open(os.path.join(tmpwin, ca_stage._MARKER), 'w') as f:
        f.write(full3)
      os.rename(tmpwin, final_path)

    r4 = ca_stage.publish(src3, parent, REL_PREFIX, _pre_rename_hook=_win_the_race)
    _ck(r4.reused and r4.dir == final3,
        'lost race to identical valid content -> reuse winner (I2, no lock)')
    # our temp must be gone (cleaned up), no .tmp left behind
    leftover = [d for d in os.listdir(parent) if '.tmp.' in d]
    _ck(not leftover, 'lost-race loser cleans up its temp (no orphan .tmp dir)')

    # --- 5. lost race to an INVALID final -> fallback, never touch final ------
    src4 = os.path.join(tmp, 'src4')
    os.makedirs(src4)
    _mk_source(src4, yml='fb: 1\n')
    full4 = sch.hash_tree(src4)
    fname4 = ca_stage.final_dirname(full4)
    final4 = os.path.join(parent, fname4)

    def _win_with_garbage(final_path):
      if os.path.exists(final_path):
        return
      os.makedirs(final_path)
      with open(os.path.join(final_path, 'BUILD'), 'w') as f:
        f.write('garbage\n')  # no marker, no config.sh -> invalid

    r5 = ca_stage.publish(src4, parent, REL_PREFIX, _pre_rename_hook=_win_with_garbage)
    _ck(r5.fallback and r5.dir != final4,
        'lost race to INVALID final -> fallback dir (correctness over cache)')
    _ck('_fb_' in os.path.basename(r5.dir), 'fallback dir carries _fb_ marker')
    # the invalid final was NOT modified by us
    _ck(os.listdir(final4) == ['BUILD'],
        'I2: invalid pre-existing final is NEVER mutated/overwritten')
    ok5, _ = ca_stage.verify_dir(r5.dir, r5.rel, r5.sha)
    _ck(ok5, 'fallback dir itself verifies (built correctly)')

    # --- 6. pre-existing INVALID final (no race) -> fallback ------------------
    src5 = os.path.join(tmp, 'src5')
    os.makedirs(src5)
    _mk_source(src5, yml='pre: 1\n')
    full5 = sch.hash_tree(src5)
    final5 = os.path.join(parent, ca_stage.final_dirname(full5))
    os.makedirs(final5)
    with open(os.path.join(final5, 'BUILD'), 'w') as f:
      f.write('truncated\n')  # invalid: no marker
    r6 = ca_stage.publish(src5, parent, REL_PREFIX)
    _ck(r6.fallback, 'pre-existing invalid final -> fallback (exists != complete, I5)')
    _ck(os.listdir(final5) == ['BUILD'], 'pre-existing invalid final untouched (I2)')

    # --- 7. truncated stage (srcfs drop-write) -> StageError, no final -------
    src6 = os.path.join(tmp, 'src6')
    os.makedirs(src6)
    _mk_source(src6, yml='trunc: 1\n')
    full6 = sch.hash_tree(src6)
    final6 = os.path.join(parent, ca_stage.final_dirname(full6))
    orig_rsync = ca_stage._rsync

    def _truncating_rsync(source, dest, timeout):
      # Simulate a drop-write: copy everything EXCEPT BUILD.
      for dirpath, _, files in os.walk(source):
        rel = os.path.relpath(dirpath, source)
        ddir = os.path.join(dest, rel) if rel != '.' else dest
        os.makedirs(ddir, exist_ok=True)
        for fn in files:
          if fn == 'BUILD':
            continue  # dropped
          shutil.copy2(os.path.join(dirpath, fn), os.path.join(ddir, fn))

    ca_stage._rsync = _truncating_rsync
    threw = False
    try:
      ca_stage.publish(src6, parent, REL_PREFIX, max_tries=2)
    except ca_stage.StageError as e:
      threw = True
      _ck(e.marker == '[[STAGE_INCOMPLETE]]', 'truncated stage raises [[STAGE_INCOMPLETE]]')
    finally:
      ca_stage._rsync = orig_rsync
    _ck(threw, 'truncated stage raises rather than publishing')
    _ck(not os.path.exists(final6),
        'I2/I4: a truncated stage NEVER creates the final dir')
    _ck(not [d for d in os.listdir(parent) if '.tmp.' in d],
        'truncated stage cleans up all its temp dirs')

    # --- 8. _safe_rmrf refuses a published dir (I3) ---------------------------
    refused = False
    try:
      ca_stage._safe_rmrf(r1.dir, parent)  # r1.dir is a published eqr_run_ca_<sha>
    except ca_stage.StageError as e:
      refused = (e.marker == '[[STAGE_RM_REFUSED]]')
    _ck(refused, 'I3: _safe_rmrf REFUSES to delete a published (non-private) dir')
    _ck(os.path.isdir(r1.dir), 'published dir still present after refused rm')
    # and it refuses something outside the parent entirely
    refused2 = False
    try:
      ca_stage._safe_rmrf(tmp, parent)
    except ca_stage.StageError:
      refused2 = True
    _ck(refused2, 'I3: _safe_rmrf refuses a path outside the staging parent')

    # --- 9. verify_dir negatives ---------------------------------------------
    ok, why = ca_stage.verify_dir(r1.dir, 'wrong/rel/eqr_run_ca_x', r1.sha)
    _ck(not ok, f'verify fails when TARGET_LABEL rel mismatches ({why})')
    ok, why = ca_stage.verify_dir(r1.dir, r1.rel, 'deadbeef' * 8)
    _ck(not ok, f'verify fails when marker sha mismatches ({why})')

    # --- 10. entry-source parsing matches a non-main.py entry ----------------
    src7 = os.path.join(tmp, 'src7')
    os.makedirs(src7)
    _mk_source(src7, yml='e: 1\n')
    # rename entry to main_eqr.py already; BUILD already says main_eqr.py. Now
    # make a BUILD whose entry is a different name and confirm verify checks it.
    with open(os.path.join(src7, 'BUILD'), 'w') as f:
      f.write('pytype_binary(\n    name = "main",\n    srcs = ["weird_entry.py"],\n)\n')
    # no weird_entry.py present -> pre-marker verify must fail -> StageError
    threw = False
    try:
      ca_stage.publish(src7, parent, REL_PREFIX, max_tries=1)
    except ca_stage.StageError:
      threw = True
    _ck(threw, 'BUILD entry-source (not hard-coded main.py) is verified present')

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
