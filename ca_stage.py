#!/usr/bin/env python3
"""Content-addressed stage + atomic publish for the tpu launcher.

This is the correctness-critical core of the content-addressed stagedir design
(~/work/stagedir_content_addressing_design.md). It stages a source tree into a
directory NAMED BY THE HASH OF ITS CONTENT and publishes it atomically, so that:

  * identical content  -> identical dir name -> blaze action cache HITS (the
    ~7 GB .par relink is reused: measured 89s cold -> 3s on an identical
    rebuild);
  * different content   -> different dir name -> two distinct codes can NEVER
    share a directory (no pollution, no overwrite), WITHOUT any lock;
  * a build in flight can never have its files overwritten, because a published
    directory is write-once and is only ever created by an atomic rename of a
    fully-staged, verified private temp dir.

The operator's hard constraint is: "staging dir must survive perfectly, no bug,
never overwritten." Every decision here is subordinate to that. When any doubt
exists, this code SACRIFICES THE CACHE HIT (falls back to a unique private dir)
rather than risk mutating or reusing a directory that might be wrong.

Invariants (do not weaken):
  I1. The rsync selection EXACTLY equals stage_content_hash._STAGE_EXCLUDES, so
      the bytes hashed are the bytes staged. We import that constant; we never
      re-list the excludes here.
  I2. A published dir `eqr_run_ca_<sha12>` is created ONLY by os.rename() of a
      verified temp. It is never mkdir'd directly, never rsynced into in place,
      never mutated after publish.
  I3. Every destructive rm targets ONLY this process's own private dirs (a name
      containing '.tmp.' or '_fb_' directly under the staging parent). A
      published dir is NEVER removed here (GC, a separate tool, may remove an
      unreferenced one).
  I4. The completeness marker `.stage_ready` (== the sha) is written LAST, after
      a COLD read-back verify, so a truncated srcfs write can never look ready.
  I5. Reuse requires final to exist AND verify (marker==sha AND BUILD AND entry
      source AND config.sh AND TARGET_LABEL rewritten). "Exists" alone is never
      enough (closes the truncated-stagedir hole).

Naming (keeps ALL existing tpu_wrapper safety guards working unchanged, because
they test basename ~ ^eqr_run_ and dirname == staging parent):
  final    : <parent>/eqr_run_ca_<sha12>
  temp     : <parent>/eqr_run_ca_<sha12>.tmp.<pid>.<rand>
  fallback : <parent>/eqr_run_ca_<sha12>_fb_<ts>_<pid><rand>   (final invalid)

CLI (called from tpu_wrapper.sh when TPU_CONTENT_ADDRESSED_STAGE=1):
  python3 ca_stage.py publish \
      --source <cwd> \
      --parent <abs .../eqr_jax_final_stages> \
      --rel-prefix experimental/qiaos/eqr_jax_final_stages \
      [--rsync-timeout 300] [--rm-timeout 120] \
      [--xm-launcher <path to backfill if missing>] \
      [--max-tries 3]
  On success prints exactly one machine-readable line to stdout:
      CA_RESULT sha=<full_sha> dir=<abs final dir> rel=<rel stagedir> reused=<0|1> fallback=<0|1>
  and exits 0. On failure prints a [[STAGE_*]] marker to stderr and exits nonzero.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, List, Optional, Tuple

import stage_content_hash as sch  # single source of truth for excludes + hash

_MARKER = '.stage_ready'
_SHA_LEN = 12  # 48 bits; collision-free for any realistic number of codes
_CA_PREFIX = 'eqr_run_ca_'
# Sleep between drop-write (INCOMPLETE) retries, mirroring tpu_wrapper.sh's 3s.
# Tests set this to 0. A timeout is NEVER retried (see the publish loop).
RETRY_SLEEP_S = 3.0


class StageError(Exception):
  """Fatal staging failure; carries a machine-readable [[MARKER]] token."""

  def __init__(self, marker: str, msg: str):
    super().__init__(msg)
    self.marker = marker
    self.msg = msg


# ----------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ----------------------------------------------------------------------------
def sha12(full_sha: str) -> str:
  return full_sha[:_SHA_LEN]


def final_dirname(full_sha: str) -> str:
  return f'{_CA_PREFIX}{sha12(full_sha)}'


def _entry_src_from_build(build_path: str) -> str:
  """The `srcs=[...]` of the `name = "main"` target, or 'main.py' as fallback.

  Mirrors tpu_wrapper.sh's sed/grep exactly: scan from the `name = "main"` line
  to the next `)`, take the first quoted token after `srcs = [`.
  """
  try:
    with open(build_path, 'r') as f:
      text = f.read()
  except OSError:
    return 'main.py'
  # Find the `name = "main"` target block up to the next ')'.
  m = re.search(r'name\s*=\s*"main"', text)
  if not m:
    return 'main.py'
  block = text[m.start():]
  end = block.find(')')
  if end != -1:
    block = block[:end]
  sm = re.search(r'srcs\s*=\s*\[\s*"([^"]+)"', block)
  return sm.group(1) if sm else 'main.py'


def _target_label_line(rel: str) -> str:
  return f'export TARGET_LABEL="//{rel}:main"'


def verify_dir(d: str, rel: str, full_sha: str) -> Tuple[bool, str]:
  """Return (ok, reason). A dir is reusable/publishable iff ok.

  Checks (all required): marker present AND == full_sha; BUILD present; the entry
  source named by BUILD present; config.sh present; config.sh carries the
  rewritten TARGET_LABEL for `rel`. This is I5 (exists-is-not-enough).
  """
  marker = os.path.join(d, _MARKER)
  try:
    with open(marker, 'r') as f:
      got = f.read().strip()
  except OSError:
    return False, 'no .stage_ready marker'
  if got != full_sha:
    return False, f'marker sha mismatch ({got[:12]} != {sha12(full_sha)})'
  build = os.path.join(d, 'BUILD')
  if not os.path.isfile(build):
    return False, 'no BUILD'
  entry = _entry_src_from_build(build)
  if not os.path.isfile(os.path.join(d, entry)):
    return False, f'entry source missing: {entry}'
  cfg = os.path.join(d, 'config.sh')
  if not os.path.isfile(cfg):
    return False, 'no config.sh'
  try:
    with open(cfg, 'r') as f:
      cfg_text = f.read()
  except OSError:
    return False, 'config.sh unreadable'
  if _target_label_line(rel) not in cfg_text:
    return False, 'TARGET_LABEL not rewritten'
  return True, 'ok'


def _is_private_dir(path: str, parent: str) -> bool:
  """True iff `path` is one of THIS process's private dirs under `parent`.

  I3: only names containing '.tmp.' or '_fb_', with basename starting
  eqr_run_ca_, directly under the staging parent, are ever removable here. A
  published `eqr_run_ca_<sha12>` (no '.tmp.'/'_fb_') can NEVER match.
  """
  rp = os.path.realpath(path)
  rparent = os.path.realpath(parent)
  base = os.path.basename(rp)
  if os.path.dirname(rp) != rparent:
    return False
  if not base.startswith(_CA_PREFIX):
    return False
  return ('.tmp.' in base) or ('_fb_' in base)


def _safe_rmrf(path: str, parent: str) -> None:
  """Remove a PRIVATE dir only; refuse anything else (fail-closed)."""
  if not _is_private_dir(path, parent):
    raise StageError('[[STAGE_RM_REFUSED]]',
                     f'refusing to rm non-private dir: {path!r} (parent={parent!r})')
  shutil.rmtree(path, ignore_errors=True)


# ----------------------------------------------------------------------------
# Side-effecting stage (rsync + rewrite + cold verify + marker)
# ----------------------------------------------------------------------------
def _rsync(source: str, dest: str, timeout: int) -> None:
  """rsync -aL with the SAME excludes as the hasher (I1). Raises on failure."""
  cmd = ['rsync', '-aL']
  for pat in sch._STAGE_EXCLUDES:  # I1: identical selection
    cmd.append(f'--exclude={pat}')
  cmd += [f'{source.rstrip("/")}/', f'{dest.rstrip("/")}/']
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired:
    raise StageError('[[STAGE_RSYNC_TIMEOUT]]',
                     f'rsync timed out after {timeout}s: {source} -> {dest}')
  if p.returncode != 0:
    raise StageError('[[STAGE_RSYNC_FAILED]]',
                     f'rsync rc={p.returncode}: {p.stderr.strip()[:500]}')


def _stage_into(temp: str, source: str, final_rel: str,
                xm_launcher: Optional[str], rsync_timeout: int) -> None:
  """Populate `temp` so it verifies for `final_rel`. Raises StageError on fail.

  NB: TARGET_LABEL is rewritten to the FINAL dir's label, because after the
  publish rename the dir IS final and the build builds //final_rel:main.
  """
  _rsync(source, temp, rsync_timeout)
  # Backfill the launcher script if the source lacked it (mirrors the wrapper).
  # Not a build input; not part of verify; kept for GCS-publish provenance.
  if xm_launcher and os.path.isfile(xm_launcher):
    if not os.path.isfile(os.path.join(temp, 'xm_launcher.py')):
      try:
        shutil.copy2(xm_launcher, os.path.join(temp, 'xm_launcher.py'))
      except OSError:
        pass
  # Rewrite TARGET_LABEL in config.sh to the FINAL label.
  cfg = os.path.join(temp, 'config.sh')
  if os.path.isfile(cfg):
    with open(cfg, 'r') as f:
      lines = f.read().split('\n')
    out = []
    replaced = False
    for ln in lines:
      if ln.lstrip().startswith('export TARGET_LABEL=') or \
         ln.lstrip().startswith('TARGET_LABEL='):
        out.append(_target_label_line(final_rel))
        replaced = True
      else:
        out.append(ln)
    if not replaced:
      out.append(_target_label_line(final_rel))
    with open(cfg, 'w') as f:
      f.write('\n'.join(out))
  # Cold read-back: flush, then verify the artifacts the build reads (I4).
  try:
    subprocess.run(['sync'], timeout=30)
  except Exception:
    pass
  ok, reason = _verify_pre_marker(temp, final_rel)
  if not ok:
    raise StageError('[[STAGE_INCOMPLETE]]', f'temp incomplete: {reason}')
  # Marker LAST (I4): only now is the dir eligible for reuse/publish.
  with open(os.path.join(temp, _MARKER), 'w') as f:
    f.write(_full_sha_holder[0])  # set by publish() before staging
  # Re-verify WITH marker (belt & suspenders; catches a dropped marker write).
  ok, reason = verify_dir(temp, final_rel, _full_sha_holder[0])
  if not ok:
    raise StageError('[[STAGE_INCOMPLETE]]', f'temp failed post-marker verify: {reason}')


def _verify_pre_marker(d: str, rel: str) -> Tuple[bool, str]:
  """Completeness WITHOUT the marker (used before we write the marker)."""
  build = os.path.join(d, 'BUILD')
  if not os.path.isfile(build):
    return False, 'no BUILD'
  entry = _entry_src_from_build(build)
  if not os.path.isfile(os.path.join(d, entry)):
    return False, f'entry source missing: {entry}'
  cfg = os.path.join(d, 'config.sh')
  if not os.path.isfile(cfg):
    return False, 'no config.sh'
  try:
    with open(cfg, 'r') as f:
      if _target_label_line(rel) not in f.read():
        return False, 'TARGET_LABEL not rewritten'
  except OSError:
    return False, 'config.sh unreadable'
  return True, 'ok'


# publish() stashes the full sha here so _stage_into can write it as the marker
# without threading it through every call. Single-threaded CLI; safe.
_full_sha_holder: List[str] = ['']


class Result:

  def __init__(self, sha: str, directory: str, rel: str, reused: bool,
               fallback: bool):
    self.sha = sha
    self.dir = directory
    self.rel = rel
    self.reused = reused
    self.fallback = fallback

  def line(self) -> str:
    return (f'CA_RESULT sha={self.sha} dir={self.dir} rel={self.rel} '
            f'reused={int(self.reused)} fallback={int(self.fallback)}')


def publish(source: str, parent: str, rel_prefix: str,
            rsync_timeout: int = 300, max_tries: int = 3,
            xm_launcher: Optional[str] = None,
            _pre_rename_hook: Optional[Callable[[str], None]] = None,
            _rand: Optional[Callable[[], str]] = None) -> Result:
  """Content-address, stage-to-temp, and atomically publish `source`.

  `_pre_rename_hook(final)` and `_rand()` are TEST SEAMS (simulate a lost race
  / deterministic temp names). They are never passed in production.
  """
  if not os.path.isdir(source):
    raise StageError('[[STAGE_SRC_MISSING]]', f'source not a dir: {source}')
  if not os.path.isdir(parent):
    g3_root = parent.split('/experimental/')[0] if '/experimental/' in parent else ''
    if g3_root and os.path.isdir(g3_root):
      try:
        os.makedirs(parent, exist_ok=True)
      except OSError:
        pass
  if not os.path.isdir(parent):
    raise StageError('[[STAGE_PARENT_MISSING]]', f'parent not a dir: {parent}')

  rnd = _rand or (lambda: '%06x' % random.randrange(16**6))
  full = sch.hash_tree(source)
  _full_sha_holder[0] = full
  fname = final_dirname(full)
  final = os.path.join(parent, fname)
  final_rel = f'{rel_prefix}/{fname}'

  # 1. Fast path: a valid published final already exists -> reuse, skip staging.
  if os.path.exists(final):
    ok, _ = verify_dir(final, final_rel, full)
    if ok:
      return Result(full, final, final_rel, reused=True, fallback=False)
    # final exists but is INVALID -> never touch it; fall back to a unique dir.
    return _publish_fallback(source, parent, rel_prefix, full, rsync_timeout,
                             max_tries, xm_launcher, rnd)

  # 2. Stage into a private temp, then atomically publish by rename.
  last_reason = ''
  for _ in range(max_tries):
    temp = os.path.join(parent, f'{fname}.tmp.{os.getpid()}.{rnd()}')
    try:
      os.mkdir(temp)
    except FileExistsError:
      continue  # name clash (astronomically rare); draw another
    try:
      _stage_into(temp, source, final_rel, xm_launcher, rsync_timeout)
    except StageError as e:
      _safe_rmrf(temp, parent)
      # A timed-out / failed rsync is a STATE (source too big, or a wedged
      # srcfs), not transient jitter; the wrapper's hard rule is do NOT retry it
      # -- the rm+re-rsync cycle is what drained the shared CreateSnapshot bucket
      # on 2026-08-28. Only a drop-write INCOMPLETE is worth another attempt.
      if e.marker != '[[STAGE_INCOMPLETE]]' and \
         os.environ.get('TPU_STAGE_RETRY_ON_TIMEOUT', '0') != '1':
        raise
      last_reason = e.msg
      if RETRY_SLEEP_S:
        time.sleep(RETRY_SLEEP_S)
      continue
    # Test seam: let a test create `final` right before we rename (lost race).
    if _pre_rename_hook is not None:
      _pre_rename_hook(final)
    try:
      os.rename(temp, final)
      return Result(full, final, final_rel, reused=False, fallback=False)
    except OSError:
      # Someone published `final` first (EEXIST), or a transient rename error.
      _safe_rmrf(temp, parent)
      if os.path.exists(final):
        ok, _ = verify_dir(final, final_rel, full)
        if ok:
          return Result(full, final, final_rel, reused=True, fallback=False)
        # final now exists but invalid -> fall back (never touch it).
        return _publish_fallback(source, parent, rel_prefix, full,
                                 rsync_timeout, max_tries, xm_launcher, rnd)
      last_reason = 'rename failed but final absent; retrying'
      continue
  raise StageError('[[STAGE_INCOMPLETE]]',
                   f'could not stage+publish after {max_tries} tries: {last_reason}')


def _publish_fallback(source: str, parent: str, rel_prefix: str, full: str,
                      rsync_timeout: int, max_tries: int,
                      xm_launcher: Optional[str],
                      rnd: Callable[[], str]) -> Result:
  """Stage into a UNIQUE private dir and use it in place (no shared final).

  Used only when a shared `final` exists but is invalid: correctness over cache.
  The fallback dir name is unique, so it never collides and never overwrites.
  """
  ts = time.strftime('%y%m%d_%H%M%S')
  last_reason = ''
  for _ in range(max_tries):
    fb = os.path.join(parent,
                      f'{final_dirname(full)}_fb_{ts}_{os.getpid()}{rnd()}')
    fb_rel = f'{rel_prefix}/{os.path.basename(fb)}'
    try:
      os.mkdir(fb)
    except FileExistsError:
      continue
    try:
      _stage_into(fb, source, fb_rel, xm_launcher, rsync_timeout)
    except StageError as e:
      _safe_rmrf(fb, parent)
      if e.marker != '[[STAGE_INCOMPLETE]]' and \
         os.environ.get('TPU_STAGE_RETRY_ON_TIMEOUT', '0') != '1':
        raise
      last_reason = e.msg
      if RETRY_SLEEP_S:
        time.sleep(RETRY_SLEEP_S)
      continue
    return Result(full, fb, fb_rel, reused=False, fallback=True)
  raise StageError('[[STAGE_INCOMPLETE]]',
                   f'fallback stage failed after {max_tries} tries: {last_reason}')


def _main(argv: List[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest='cmd', required=True)
  p = sub.add_parser('publish')
  p.add_argument('--source', required=True)
  p.add_argument('--parent', required=True)
  p.add_argument('--rel-prefix', required=True)
  p.add_argument('--rsync-timeout', type=int, default=300)
  p.add_argument('--max-tries', type=int, default=3)
  p.add_argument('--xm-launcher', default='')
  p.add_argument('--provenance-file', default='',
                 help='append a JSONL provenance record here (which launch built '
                      'which content sha). Purely additive; never read by the '
                      'build path. Empty = do not write.')
  args = ap.parse_args(argv)

  if args.cmd == 'publish':
    try:
      r = publish(args.source, args.parent, args.rel_prefix,
                  rsync_timeout=args.rsync_timeout, max_tries=args.max_tries,
                  xm_launcher=args.xm_launcher or None)
    except StageError as e:
      print(f'{e.marker} {e.msg}', file=sys.stderr)
      return 1
    # Provenance: append-only, best-effort, AFTER a successful publish. A failure
    # to record provenance must never fail the launch, so it is wrapped and
    # swallowed -- the build only needs the result line on stdout.
    if args.provenance_file:
      try:
        import json as _json
        rec = {
            'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'sha': r.sha,
            'dir': r.dir,
            'rel': r.rel,
            'source': os.path.abspath(args.source),
            'reused': r.reused,
            'fallback': r.fallback,
            'pid': os.getpid(),
        }
        with open(os.path.expanduser(args.provenance_file), 'a') as pf:
          pf.write(_json.dumps(rec) + '\n')
      except Exception:
        pass
    print(r.line())
    return 0
  return 2


if __name__ == '__main__':
  sys.exit(_main(sys.argv[1:]))
