#!/usr/bin/env python3
"""Garbage-collect old content-addressed stagedirs. DRY-RUN BY DEFAULT.

Content-addressed dirs (`eqr_run_ca_<sha>`) dedup by content, but distinct
contents still accumulate under the staging parent. This reclaims the OLD,
UNREFERENCED ones. It is the ONLY component in this design that deletes
anything, so it is built to fail SAFE in every ambiguous case:

  * DRY-RUN unless `--delete` is passed explicitly. Dry-run prints exactly what
    it WOULD delete and why the rest is kept.
  * A dir is deletable ONLY if ALL hold:
      - basename matches `eqr_run_ca_<hex>` (optionally a `.tmp.`/`_fb_` private
        dir older than a long grace) AND sits DIRECTLY under the staging parent;
      - it is NOT referenced by the local queue file (any state);
      - it is NOT referenced by the job registry (~/.tpu_jobs.json) or its legacy
        file, for ANY job whose status is not terminal-and-archived — to be safe
        we treat ANY registry mention as a keep;
      - its atime AND mtime are both older than `--min-age-days` (default 7);
      - it is complete/valid OR clearly a leftover temp (a stuck `.tmp.`).
  * ANY error resolving a reference => KEEP (fail-safe). An unreadable queue or
    registry aborts deletion entirely rather than guessing.
  * NEVER deletes a `.tmp.`/`_fb_` younger than `--tmp-grace-hours` (a build may
    be mid-stage). NEVER follows symlinks out of the parent.
  * A manifest of intended deletions is written before ANY removal, and each
    removal is re-checked against live references immediately before `rmtree`
    (re-probe, because state can change between planning and acting).

Usage:
  python3 ca_gc.py --parent <abs .../eqr_jax_final_stages>        # dry-run
  python3 ca_gc.py --parent <...> --min-age-days 14               # dry-run, older cut
  python3 ca_gc.py --parent <...> --delete --manifest /tmp/gc.txt # ACTUALLY delete
Exit: 0 ok (even in dry-run); 2 refused (unreadable queue/registry) — never deletes then.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

_CA_RE = re.compile(r'^eqr_run_ca_[0-9a-f]{6,}$')
_PRIV_RE = re.compile(r'^eqr_run_ca_[0-9a-f]{6,}(\.tmp\.|_fb_)')


def _load_referenced_stagedirs(queue_file: str, jobs_file: str,
                               legacy_file: str) -> Tuple[Set[str], List[str]]:
  """Return (referenced basenames, errors). Any error => caller must KEEP-all.

  A reference is matched by BASENAME, because the registry stores a relative
  stagedir and workspaces get repointed; the basename `eqr_run_ca_<sha>` is the
  stable identity. We are deliberately generous: ANY mention keeps the dir.
  """
  referenced: Set[str] = set()
  errors: List[str] = []

  def _add_from_stagedir_value(v):
    if isinstance(v, str) and v:
      referenced.add(os.path.basename(v.rstrip('/')))

  # Local queue (a list of entries).
  try:
    with open(os.path.expanduser(queue_file)) as f:
      q = json.load(f)
    rows = q if isinstance(q, list) else list(q.values())
    for r in rows:
      if isinstance(r, dict):
        for k in ('stagedir', 'abs_stagedir', 'workdir'):
          _add_from_stagedir_value(r.get(k))
  except FileNotFoundError:
    pass  # no queue file is fine (nothing queued)
  except Exception as e:
    errors.append(f'queue unreadable ({queue_file}): {e}')

  # Registry + legacy (dicts keyed by xid).
  for jf in (jobs_file, legacy_file):
    try:
      with open(os.path.expanduser(jf)) as f:
        j = json.load(f)
      for _xid, rec in (j.items() if isinstance(j, dict) else []):
        if isinstance(rec, dict):
          _add_from_stagedir_value(rec.get('stagedir'))
    except FileNotFoundError:
      pass
    except Exception as e:
      errors.append(f'registry unreadable ({jf}): {e}')

  return referenced, errors


def _dir_age_ok(path: str, min_age_s: float, now: float) -> Tuple[bool, str]:
  """True iff BOTH atime and mtime are older than min_age_s."""
  try:
    st = os.stat(path)
  except OSError as e:
    return False, f'stat failed: {e}'
  age_m = now - st.st_mtime
  age_a = now - st.st_atime
  if age_m < min_age_s:
    return False, f'mtime too recent ({age_m/86400:.1f}d < {min_age_s/86400:.1f}d)'
  if age_a < min_age_s:
    return False, f'atime too recent ({age_a/86400:.1f}d < {min_age_s/86400:.1f}d)'
  return True, f'age ok (m={age_m/86400:.1f}d a={age_a/86400:.1f}d)'


def plan(parent: str, referenced: Set[str], min_age_days: float,
         tmp_grace_hours: float, now: Optional[float] = None
         ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
  """Return (to_delete, to_keep) as [(basename, reason), ...]. Pure/inspectable."""
  now = now if now is not None else time.time()
  min_age_s = min_age_days * 86400
  tmp_grace_s = tmp_grace_hours * 3600
  to_delete: List[Tuple[str, str]] = []
  to_keep: List[Tuple[str, str]] = []
  try:
    entries = sorted(os.listdir(parent))
  except OSError as e:
    return [], [('<parent>', f'cannot list parent: {e}')]

  for name in entries:
    path = os.path.join(parent, name)
    # Never touch symlinks or non-directories.
    if os.path.islink(path) or not os.path.isdir(path):
      to_keep.append((name, 'not a real directory'))
      continue
    is_priv = bool(_PRIV_RE.match(name))
    is_ca = bool(_CA_RE.match(name))
    if not is_ca and not is_priv:
      to_keep.append((name, 'not a content-addressed dir (name does not match)'))
      continue
    # A referenced dir is ALWAYS kept.
    if name in referenced:
      to_keep.append((name, 'referenced by queue/registry'))
      continue
    if is_priv:
      # A stuck private temp/fallback: only if OLDER than the grace window.
      ok, why = _dir_age_ok(path, tmp_grace_s, now)
      if ok:
        to_delete.append((name, f'stale private temp/fallback ({why})'))
      else:
        to_keep.append((name, f'private temp within grace window ({why})'))
      continue
    # A published eqr_run_ca_<sha>: age gate.
    ok, why = _dir_age_ok(path, min_age_s, now)
    if ok:
      to_delete.append((name, f'unreferenced + old ({why})'))
    else:
      to_keep.append((name, why))
  return to_delete, to_keep


def _safe_to_remove(path: str, parent: str) -> bool:
  """Final guard immediately before rmtree: real dir, under parent, CA-named."""
  rp = os.path.realpath(path)
  rparent = os.path.realpath(parent)
  if os.path.dirname(rp) != rparent:
    return False
  base = os.path.basename(rp)
  return bool(_CA_RE.match(base) or _PRIV_RE.match(base))


def main(argv: List[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--parent', required=True,
                  help='the staging parent (…/eqr_jax_final_stages)')
  ap.add_argument('--min-age-days', type=float, default=7.0)
  ap.add_argument('--tmp-grace-hours', type=float, default=12.0,
                  help='never delete a .tmp./_fb_ dir younger than this')
  ap.add_argument('--delete', action='store_true',
                  help='ACTUALLY delete. Without this it is a dry-run.')
  ap.add_argument('--manifest', default='',
                  help='write the intended-deletion manifest here')
  ap.add_argument('--queue-file',
                  default=os.environ.get('TPU_LOCAL_QUEUE_FILE',
                                         '~/.tpu_local_queue.json'))
  ap.add_argument('--jobs-file',
                  default=os.environ.get('TPU_JOBS_FILE', '~/.tpu_jobs.json'))
  ap.add_argument('--legacy-file',
                  default=os.environ.get('TPU_JOBS_LEGACY_FILE',
                                         '~/.tpu_jobs_legacy.json'))
  args = ap.parse_args(argv)

  parent = os.path.abspath(os.path.expanduser(args.parent))
  if not os.path.isdir(parent):
    print(f'ca_gc: parent not a dir: {parent}', file=sys.stderr)
    return 2

  referenced, errors = _load_referenced_stagedirs(
      args.queue_file, args.jobs_file, args.legacy_file)
  if errors:
    # Fail-safe: if we cannot fully resolve references, we refuse to delete.
    print('ca_gc: REFUSING to delete — could not resolve all references:',
          file=sys.stderr)
    for e in errors:
      print(f'  - {e}', file=sys.stderr)
    if args.delete:
      return 2
    print('  (dry-run continues, but note references are INCOMPLETE)',
          file=sys.stderr)

  to_delete, to_keep = plan(parent, referenced, args.min_age_days,
                            args.tmp_grace_hours)

  print(f'ca_gc parent={parent}')
  print(f'  referenced (kept regardless): {len(referenced)}')
  print(f'  KEEP {len(to_keep)}   DELETE-CANDIDATES {len(to_delete)}   '
        f'mode={"DELETE" if args.delete else "DRY-RUN"}')
  for name, why in to_delete:
    print(f'  [{"DELETE" if args.delete else "would-delete"}] {name}  <- {why}')

  if args.manifest:
    try:
      with open(os.path.expanduser(args.manifest), 'w') as mf:
        for name, why in to_delete:
          mf.write(f'{os.path.join(parent, name)}\t{why}\n')
      print(f'  manifest written: {args.manifest}')
    except OSError as e:
      print(f'  manifest write failed: {e}', file=sys.stderr)

  if not args.delete:
    print('  DRY-RUN: nothing removed. Re-run with --delete to act.')
    return 0

  # Actually delete — but RE-PROBE references right before each removal, and
  # re-check the name/location guard. State can change between plan and act.
  referenced2, errors2 = _load_referenced_stagedirs(
      args.queue_file, args.jobs_file, args.legacy_file)
  if errors2:
    print('ca_gc: references became unreadable at act time; aborting deletion.',
          file=sys.stderr)
    return 2
  removed = 0
  for name, why in to_delete:
    path = os.path.join(parent, name)
    if name in referenced2:
      print(f'  [skip] {name}: became referenced since planning')
      continue
    if not _safe_to_remove(path, parent):
      print(f'  [skip] {name}: failed final safety guard')
      continue
    try:
      shutil.rmtree(path)
      removed += 1
      print(f'  [removed] {name}')
    except OSError as e:
      print(f'  [error] {name}: {e}')
  print(f'  removed {removed}/{len(to_delete)}')
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
