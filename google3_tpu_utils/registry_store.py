"""The ONE write path for the XID registry (`~/.tpu_jobs.json`) and its archive.

Every process that changes the registry -- the wrapper (register / corpse
removal / cancel), xm_launcher, the check daemon, `tpu clear`, the router's
reroute archive, budget_enforcer -- goes through `mutate` / `archive` /
`patch_fields` here. Nobody opens the registry for writing on their own.

THE PROTOCOL (what this module guarantees, and what a new writer must not
re-invent):

  R1  The exclusive flock on the registry FILE ITSELF is held across the whole
      read-modify-write. Never truncate before holding it.
  R2  Strict read. A missing file is an empty registry; a file that exists but
      is empty or not a JSON object is UNREADABLE. Retry (releasing the lock in
      between, so a writer that truncated before locking can finish), then
      raise. Never treat "could not read" as "empty": that is how one writer's
      truncation window became every entry lost.
  R3  Serialize fully before touching the file, then write in place and
      truncate after writing, flush + fsync, all under the lock.
  R4  The archive (legacy) file is written only while holding the registry
      lock, atomically (tmp + fsync + rename). An archive writes the archive
      FIRST and the registry SECOND: a crash in between leaves an entry in both
      files (harmless), never in neither.
  R5  Every change appends one line to `<registry>.journal`; every successful
      write refreshes `<registry>.lastgood`. A lost entry is then traceable to
      the process that removed it, and a corrupt registry is recoverable.
  R6  Readers take a shared flock (`read`): writes are in place, so a lock-less
      reader can see a half-written file.

WHY IN PLACE AND NOT TMP+RENAME for the registry: long-lived processes that
still run older code lock the registry's inode. A rename would hand them a lock
on an orphaned inode and their writes would vanish. Once every writer goes
through this module the registry can move to a sidecar lock + rename.

Standard library only: it is imported by bash-embedded python, by xm_launcher
under the xmanager interpreter, by plain-python daemons and by blaze binaries.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import json
import os
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Optional

DEFAULT_JOBS_FILE = '~/.tpu_jobs.json'
READ_ATTEMPTS = 25        # x READ_RETRY_S ~= 5 s for a truncating writer to finish
READ_RETRY_S = 0.2
JOURNAL_MAX_BYTES = 50 * 1024 * 1024
_ABSENT = object()


class RegistryUnreadable(RuntimeError):
  """The file exists but does not hold a JSON object, even after retries.

  Nothing is written when this is raised. Recover with
  `python3 registry_store.py repair` (restores `<registry>.lastgood`)."""


# --- paths -------------------------------------------------------------------
def jobs_path(path: Optional[str] = None) -> str:
  """Explicit path > $TPU_JOBS_FILE > ~/.tpu_jobs.json."""
  return os.path.abspath(os.path.expanduser(
      path or os.environ.get('TPU_JOBS_FILE') or DEFAULT_JOBS_FILE))


def legacy_path(legacy: Optional[str] = None, jobs: Optional[str] = None) -> str:
  """Explicit path > $TPU_JOBS_LEGACY_FILE > derived from the registry path.

  Derivation maps `X.json` to `X_legacy.json`, which is exactly the pairing of
  both lanes (~/.tpu_jobs.json -> ~/.tpu_jobs_legacy.json,
  ~/lyy-work/.npu_jobs.json -> ~/lyy-work/.npu_jobs_legacy.json)."""
  if legacy:
    return os.path.abspath(os.path.expanduser(legacy))
  env = os.environ.get('TPU_JOBS_LEGACY_FILE')
  if env:
    return os.path.abspath(os.path.expanduser(env))
  j = jobs_path(jobs)
  root, ext = os.path.splitext(j)
  return f'{root}_legacy{ext or ".json"}'


# --- strict parsing ------------------------------------------------------------
def _parse(text: str, path: str) -> dict:
  if not text.strip():
    raise ValueError(f'{path} is empty')
  data = json.loads(text)
  if not isinstance(data, dict):
    raise ValueError(f'{path} holds a {type(data).__name__}, not an object')
  return data


def _read_with_retries(fh, path: str, lock_kind: int, attempts: Optional[int],
                       retry_s: Optional[float]) -> dict:
  """Acquire `lock_kind` on fh and return the parsed file, HOLDING the lock.

  On an unreadable file the lock is RELEASED before sleeping: a writer running
  older code may have truncated the file and be waiting for this very lock to
  write its content. Holding it while retrying would starve that writer."""
  attempts = READ_ATTEMPTS if attempts is None else attempts
  retry_s = READ_RETRY_S if retry_s is None else retry_s
  last_err: Optional[Exception] = None
  for i in range(max(1, attempts)):
    fcntl.flock(fh, lock_kind)
    fh.seek(0)
    text = fh.read()
    try:
      return _parse(text, path)
    except ValueError as e:
      last_err = e
      fcntl.flock(fh, fcntl.LOCK_UN)
      if i + 1 < attempts:
        time.sleep(retry_s)
  raise RegistryUnreadable(
      f'{path}: unreadable after {attempts} attempts ({last_err}); refusing to '
      f'treat it as empty. Restore with: python3 {os.path.abspath(__file__)} '
      f'repair --jobs_file={path}')


def read(path: Optional[str] = None, *, attempts: Optional[int] = None,
         retry_s: Optional[float] = None) -> dict:
  """Strict read under a shared lock. Missing file -> {}; unreadable -> raise."""
  p = jobs_path(path)
  if not os.path.exists(p):
    return {}
  with open(p, 'r', encoding='utf-8') as fh:
    try:
      return _read_with_retries(fh, p, fcntl.LOCK_SH, attempts, retry_s)
    finally:
      fcntl.flock(fh, fcntl.LOCK_UN)


def read_legacy(legacy: Optional[str] = None, jobs: Optional[str] = None, *,
                attempts: Optional[int] = None,
                retry_s: Optional[float] = None) -> dict:
  """Strict read of the archive. It is replaced atomically, so no lock needed;
  retries only cover a writer still running older (truncating) code."""
  p = legacy_path(legacy, jobs)
  if not os.path.exists(p):
    return {}
  attempts = READ_ATTEMPTS if attempts is None else attempts
  retry_s = READ_RETRY_S if retry_s is None else retry_s
  last_err: Optional[Exception] = None
  for i in range(max(1, attempts)):
    try:
      with open(p, 'r', encoding='utf-8') as fh:
        return _parse(fh.read(), p)
    except ValueError as e:
      last_err = e
      if i + 1 < attempts:
        time.sleep(retry_s)
  raise RegistryUnreadable(f'{p}: unreadable after {attempts} attempts ({last_err})')


# --- journal / lastgood --------------------------------------------------------
def _short(v: Any, n: int = 160) -> Any:
  s = json.dumps(v, default=str)
  return v if len(s) <= n else s[:n] + '...'


def _diff(before: dict, after: dict) -> dict:
  added = sorted(set(after) - set(before))
  removed = sorted(set(before) - set(after))
  changed = {}
  for k in set(before) & set(after):
    b, a = before[k], after[k]
    if b == a:
      continue
    if isinstance(b, dict) and isinstance(a, dict):
      changed[k] = {f: [_short(b.get(f)), _short(a.get(f))]
                    for f in sorted(set(b) | set(a)) if b.get(f) != a.get(f)}
    else:
      changed[k] = [_short(b), _short(a)]
  return {'added': added, 'removed': removed, 'changed': changed}


def _journal(path: str, who: str, record: dict) -> None:
  """Best-effort append; a journal failure never fails the write it records."""
  jp = path + '.journal'
  try:
    if os.path.exists(jp) and os.path.getsize(jp) > JOURNAL_MAX_BYTES:
      os.replace(jp, jp + '.1')
    line = json.dumps({
        'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'who': who, 'pid': os.getpid(), 'argv0': os.path.basename(sys.argv[0] if sys.argv else ''),
        **record}, default=str, sort_keys=True)
    fd = os.open(jp, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
      os.write(fd, (line + '\n').encode('utf-8'))
    finally:
      os.close(fd)
  except OSError:
    pass


def _atomic_write(path: str, payload: str) -> None:
  """tmp in the same directory + fsync + rename; keeps the old file's mode."""
  d = os.path.dirname(path) or '.'
  try:
    mode = os.stat(path).st_mode & 0o777
  except OSError:
    mode = 0o640
  fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.tmp.')
  try:
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
      fh.write(payload)
      fh.flush()
      os.fsync(fh.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)
  except BaseException:
    with contextlib.suppress(OSError):
      os.unlink(tmp)
    raise


# --- the write primitive ---------------------------------------------------------
def mutate(fn: Callable[[dict], Any], *, who: str, path: Optional[str] = None,
           create: bool = True, attempts: Optional[int] = None,
           retry_s: Optional[float] = None,
           _after_write: Optional[Callable[[], None]] = None) -> Any:
  """Run fn(data) on the live registry under the exclusive lock; persist if it
  changed anything. Returns fn's return value.

  `fn` mutates the dict in place and must be quick (no RPCs, no builds): the
  lock is held while it runs. `create=False` makes a missing registry a no-op
  (fn is not called, None is returned)."""
  p = jobs_path(path)
  existed = os.path.exists(p)
  if not existed and not create:
    return None
  fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o640)
  with os.fdopen(fd, 'r+', encoding='utf-8') as fh:
    if not existed and os.fstat(fh.fileno()).st_size == 0:
      fcntl.flock(fh, fcntl.LOCK_EX)
      fh.seek(0)
      text = fh.read()
      data = _parse(text, p) if text.strip() else {}
    else:
      data = _read_with_retries(fh, p, fcntl.LOCK_EX, attempts, retry_s)
    try:
      before = json.loads(json.dumps(data))
      result = fn(data)
      unchanged = (json.dumps(data, sort_keys=True)
                   == json.dumps(before, sort_keys=True))
      if unchanged and existed:
        return result
      # A registry this call created is always written, so it never stays a
      # 0-byte file that every strict reader would refuse.
      payload = json.dumps(data, indent=2)
      fh.seek(0)
      fh.write(payload)
      fh.truncate()
      fh.flush()
      os.fsync(fh.fileno())
      if _after_write is not None:
        _after_write()
      _journal(p, who, _diff(before, data))
      with contextlib.suppress(OSError):
        _atomic_write(p + '.lastgood', payload)
      return result
    finally:
      fcntl.flock(fh, fcntl.LOCK_UN)


def patch_fields(patches: dict, *, who: str, path: Optional[str] = None) -> dict:
  """Compare-and-set field updates computed from an earlier `read` snapshot.

  patches = {xid: {field: (expected_old, new)}}. A field is written only if its
  live value still equals expected_old (use `ABSENT` for "was not set"), so a
  change made by someone else since the snapshot -- e.g. a `tpu cancel` that
  set CANCELLED while a status pass was thinking -- is never overwritten.
  Entries that disappeared are skipped. Returns {'applied': n, 'conflicts': [..]}."""
  out = {'applied': 0, 'conflicts': []}

  def _apply(data: dict) -> None:
    for xid, fields in patches.items():
      entry = data.get(xid)
      if not isinstance(entry, dict):
        out['conflicts'].append((xid, '*', 'entry gone'))
        continue
      for field, (expected, new) in fields.items():
        live = entry.get(field, _ABSENT)
        if live != expected and not (live is _ABSENT and expected is _ABSENT):
          out['conflicts'].append((xid, field, f'live={live!r} expected={expected!r}'))
          continue
        if new is _ABSENT:
          entry.pop(field, None)
        else:
          entry[field] = new
        out['applied'] += 1

  mutate(_apply, who=who, path=path, create=False)
  return out


ABSENT = _ABSENT
_EMPTY_VALUES = (None, '', 0, {}, [])


def _is_empty(entry: dict, field: str) -> bool:
  return field not in entry or entry[field] in _EMPTY_VALUES


def register(xid: str, *, who: str, overwrite: Optional[dict] = None,
             fill: Optional[dict] = None, path: Optional[str] = None) -> dict:
  """Create or update one XID's entry. `overwrite` fields are always set;
  `fill` fields only where the entry has no value yet (missing, None, '', 0,
  {} or []), so a later writer never erases what an earlier one recorded.
  Returns the entry as written."""
  xid = str(xid)

  def _do(data: dict) -> dict:
    entry = data.get(xid)
    if not isinstance(entry, dict):
      entry = {}
    entry.update(overwrite or {})
    for k, v in (fill or {}).items():
      if _is_empty(entry, k):
        entry[k] = v
    data[xid] = entry
    return dict(entry)

  return mutate(_do, who=who, path=path)


def mark_cancelled(xids: Iterable[str], *, who: str,
                   path: Optional[str] = None) -> list:
  """Record a deliberate stop on each XID that has an entry. Returns the XIDs
  marked. retry_count is pinned so no resubmit path can resurrect the job."""
  xids = [str(x) for x in xids]
  now = time.strftime('%Y-%m-%d %H:%M:%S')

  def _do(data: dict) -> list:
    done = []
    for x in xids:
      entry = data.get(x)
      if not isinstance(entry, dict):
        continue
      entry.update({'status': 'CANCELLED', 'error': '', 'cancelled_at': now,
                    'retry_count': 5})
      done.append(x)
    return done

  return mutate(_do, who=who, path=path, create=False) or []


def remove_corpse(xid: str, *, who: str, path: Optional[str] = None) -> bool:
  """Drop an entry a launcher pre-registered and never finished: one with no
  tier/alloc/status. A healthy entry is never touched. Returns True if removed."""
  xid = str(xid)

  def _do(data: dict) -> bool:
    entry = data.get(xid)
    if isinstance(entry, dict) and not any(k in entry for k in ('tier', 'alloc', 'status')):
      del data[xid]
      return True
    return False

  return bool(mutate(_do, who=who, path=path, create=False))


def archive(xids: Iterable[str], *, who: str, archived_by: str,
            extra: Optional[dict] = None, defaults: Optional[dict] = None,
            path: Optional[str] = None, legacy: Optional[str] = None,
            create: bool = False) -> dict:
  """Move registry entries into the archive (legacy) file. Archive first,
  registry second (R4). Idempotent: an xid already off the registry is a no-op
  unless `extra` carries data for it, which is merged into its archive record.

  extra = {xid: {field: value}} overwrites fields of each archive record (e.g.
  the local-queue row `tpu clear` folds in); defaults = {xid: {field: value}}
  only fills fields the record does not have yet. Returns
  {'moved': [...], 'merged': [...]}. Raises (and writes nothing) if either
  file is unreadable."""
  xids = [str(x) for x in xids if str(x)]
  extra = {str(k): v for k, v in (extra or {}).items()}
  defaults = {str(k): v for k, v in (defaults or {}).items()}
  lp = legacy_path(legacy, path)
  out = {'moved': [], 'merged': []}

  def _do(data: dict) -> None:
    targets = [x for x in dict.fromkeys(xids + list(extra) + list(defaults))
               if x in data or extra.get(x) or defaults.get(x)]
    if not targets:
      return
    leg = read_legacy(lp)
    now = datetime.datetime.now().isoformat(timespec='seconds')
    for x in targets:
      rec = dict(leg.get(x) or {})
      if x in data:
        rec.update(data[x] if isinstance(data[x], dict) else {'value': data[x]})
        rec['archived_at'] = now
        rec['archived_by'] = archived_by
        out['moved'].append(x)
      else:
        rec.setdefault('archived_at', now)
        rec.setdefault('archived_by', archived_by)
        out['merged'].append(x)
      rec.update(extra.get(x) or {})
      for k, v in (defaults.get(x) or {}).items():
        rec.setdefault(k, v)
      leg[x] = rec
    _atomic_write(lp, json.dumps(leg, indent=2, sort_keys=True))   # archive FIRST
    for x in out['moved']:
      data.pop(x, None)                                             # registry SECOND

  mutate(_do, who=f'{who} (archive -> {os.path.basename(lp)})', path=path,
         create=create)
  if out['merged'] and not out['moved']:
    _journal(jobs_path(path), who, {'archive_merged_only': out['merged']})
  return out


def stash_unwritten(record: dict, *, who: str, path: Optional[str] = None) -> str:
  """Append a registry change that could NOT be applied (registry unreadable)
  to `<registry>.unwritten.jsonl`, so a launched job's metadata is never just
  dropped. Returns the stash path. Best-effort; never raises."""
  sp = jobs_path(path) + '.unwritten.jsonl'
  with contextlib.suppress(OSError):
    line = json.dumps({'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds'), 'who': who, 'pid': os.getpid(), **record},
                      default=str, sort_keys=True)
    fd = os.open(sp, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
      os.write(fd, (line + '\n').encode('utf-8'))
    finally:
      os.close(fd)
  return sp


# --- repair ---------------------------------------------------------------------
def repair(path: Optional[str] = None) -> str:
  """Restore `<registry>.lastgood` into the registry, ONLY if the registry is
  unreadable, under the exclusive lock. Returns a one-line account.

  Entries written after the last successful write through this module are
  lost; that is the price of a writer outside this module having corrupted the
  file, and the journal names what the restore dropped."""
  p = jobs_path(path)
  lg = p + '.lastgood'
  try:
    read(p, attempts=3, retry_s=0.1)
    return f'{p} is readable; nothing to repair.'
  except RegistryUnreadable:
    pass
  if not os.path.exists(lg):
    return f'{p} is unreadable and there is no {lg} to restore from.'
  with open(lg, encoding='utf-8') as fh:
    good = _parse(fh.read(), lg)
  with open(p, 'r+', encoding='utf-8') as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
      fh.seek(0)
      try:
        _parse(fh.read(), p)
        return f'{p} became readable meanwhile; not touching it.'
      except ValueError:
        pass
      fh.seek(0)
      fh.write(json.dumps(good, indent=2))
      fh.truncate()
      fh.flush()
      os.fsync(fh.fileno())
    finally:
      fcntl.flock(fh, fcntl.LOCK_UN)
  _journal(p, 'registry_store repair', {'restored_from': lg, 'entries': len(good)})
  return f'restored {len(good)} entries into {p} from {lg}'


# --- CLI: verify / repair / journal / cancel ---------------------------------------
def _cli(argv: list[str]) -> int:
  import argparse
  ap = argparse.ArgumentParser(prog='registry_store')
  ap.add_argument('cmd', choices=['verify', 'repair', 'journal', 'cancel'])
  ap.add_argument('xids', nargs='*', help='for `cancel`')
  ap.add_argument('--jobs_file', default=None)
  ap.add_argument('--legacy_file', default=None)
  ap.add_argument('--who', default='cli')
  ap.add_argument('-n', type=int, default=20)
  a = ap.parse_args(argv)
  p = jobs_path(a.jobs_file)
  if a.cmd == 'cancel':
    done = mark_cancelled(a.xids, who=a.who, path=p)
    print(f'marked CANCELLED in {p}: {" ".join(done) if done else "(none on the registry)"}')
    return 0
  if a.cmd == 'verify':
    rc = 0
    for label, fn in (('registry', lambda: read(p, attempts=3)),
                      ('archive', lambda: read_legacy(a.legacy_file, p, attempts=3))):
      try:
        print(f'{label}: OK, {len(fn())} entries')
      except RegistryUnreadable as e:
        print(f'{label}: UNREADABLE: {e}')
        rc = 1
    return rc
  if a.cmd == 'journal':
    jp = p + '.journal'
    if os.path.exists(jp):
      with open(jp, encoding='utf-8') as fh:
        for line in fh.readlines()[-a.n:]:
          print(line.rstrip())
    return 0
  msg = repair(p)
  print(msg)
  return 1 if 'no ' in msg and 'to restore' in msg else 0


if __name__ == '__main__':
  sys.exit(_cli(sys.argv[1:]))
