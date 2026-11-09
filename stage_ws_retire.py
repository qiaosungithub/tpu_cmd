#!/usr/bin/env python3
"""Retire full, idle auto-rotated staging workspaces. REPORT-ONLY BY DEFAULT.

WHY. stage_ws_rotate.py only ever ADDS CitC clients (one per operator family
every ~5-6 days), so the client count grows without bound. Deleting a client
destroys the stagedirs inside it, and `tpu queue --resume_xid=X` re-runs the
stagedir recorded for X, so a careless delete breaks resumes. This tool decides
which clients could go, and (only with an explicit --execute, which nothing
calls automatically) archives every stagedir an old record still names to CNS
BEFORE deleting the client, with an XID -> tarball index and a restore command.

POLICY. Candidates are ONLY auto-created family members (the strict name regex
of stage_ws_rotate.FAMILIES: tpu_staging_t215_<YYYYMMDD>[_n] for qiaos,
tpu_staging_lyy_t215_<YYYYMMDD>[_n] for lyy). Every other client is listed as
"manual, not managed" and never considered. A member is RETIRABLE only if ALL
of these hold (an unknown value never counts as holding):
  1. it is not the newest HEALTHY member of its family (when no member is
     healthy, the newest member is protected instead);
  2. its .citc/manifest is >= 9 MiB (TPU_STAGE_WS_RETIRE_MIN_MANIFEST): it has
     rotated out;
  3. its last stage is more than 30 days ago (TPU_STAGE_WS_RETIRE_IDLE_DAYS):
     the newest of the staging parent's mtime, every stagedir's mtime, and every
     provenance record whose `dir` is inside it (a CA re-use leaves no mtime);
  4. no stagedir inside it is referenced by
       - a local-queue entry or an active-registry entry (either operator)
         that still BLOCKS. An entry stops blocking only when BOTH its status
         is terminal AND it ended more than 30 days ago
         (TPU_STAGE_WS_RETIRE_ENDED_DAYS). Everything else blocks: a live,
         unknown or missing status, a terminal entry with no end timestamp,
         and one that ended <= 30 days ago. Terminal values, from the code
         that writes them:
           queue `state`      DONE, FAILED -- route_lib.FINISHED_STATES, the
                              queue's one definition of "the run ended"
                              (RUNNING is not finished there either);
           registry `status`  COMPLETED, FAILED (tpu_check_daemon.sh status
                              pass), CANCELLED (registry_store.cancel, with
                              cancelled_at), TERMINAL_RECONCILED and
                              TERMINAL_NOT_RUNNING (~/lyy-work/
                              reconcile_finished.py). SUBMITTED / RUNNING are
                              live.
         End timestamp of a queue entry: the newest *_at in it, and only if
         its final attempt (submissions[], highest seq) has `ended_at`
         (route_lib end_attempt). Of a registry row: the newest of its own
         *_at fields (cancelled_at), the `ended_at` the queue recorded for the
         same XID, and the time of the newest <registry>.journal line about
         that XID -- the journal counts only when its newest `status` change
         for the XID set the row's CURRENT status (a status written by a
         non-journaling writer, e.g. reconcile_finished.py, happened at an
         unknown time). Registry rows carry no other timestamp.
         Every entry, blocking or not, is followed through each XID it lists
         (xid, prior_xids, submissions[].xid, resolved via every registry),
         so a running or recent resume keeps its old terminal XIDs' stagedirs
         blocked;
       - any other registry or provenance record (legacy registries,
         ~/.tpu_ca_provenance.jsonl, ...) younger than 30 days
         (TPU_STAGE_WS_RETIRE_RECORD_DAYS). A record whose age cannot be
         established counts as young.
     A RELATIVE stagedir record matches by basename in every client (the
     wrapper's --resume_xid search tries every workspace); an ABSOLUTE one
     matches only its own client. Any record source that exists but cannot be
     read makes the whole plan INCOMPLETE, and then nothing is retirable.
  These are the same sources ca_gc.py consults (queue, ~/.tpu_jobs.json,
  ~/.tpu_jobs_legacy.json), plus lyy's copies, the provenance log and the older
  registries.

EXECUTE (`retire --execute [--client NAME ...]`; never wired anywhere). Per
retirable client, in this order, stopping at the first failure:
  1. tar+gzip every stagedir that ANY record references (including terminal
     entries that ended > 30 days ago and no longer block), list it back,
     sha256 it (in ~/tmp, not the RAM /tmp);
  2. copy each tarball to <archive root>/<client>/<stagedir>.tar.gz with
     fileutil, one copy at a time, each call gated by a wait loop that sleeps
     while `pgrep -c fileutil` > 12; then copy it BACK and require the same
     sha256;
  3. write INDEX.jsonl (one row per XID: XID -> tarball, sha256, client,
     baseline CL, stagedir), upload and verify it the same way, and append the
     rows to the local index ~/.tpu_stage_ws_retired_index.jsonl;
  4. holding the wrapper's stage lock (/tmp/tpu_stage.<uid>.lock, so no stage or
     resume runs meanwhile), re-plan from fresh records; only if the client is
     still retirable with the same stagedirs, `g4 citc -d <client>` (never -f).
  If any of 1-3 fails, nothing is deleted and the exit code is 1.
  Archive root: <metro storage>/home/qiaos/eqr_data/stage_ws_retired, the
  metro's group storage cell resolved by google3_tpu_utils/cell_locality.py
  (never a hand-written table). The workstation is in us-east4 (metro iad, no
  group storage), so the default metro is the nearest one with storage, cmh
  (-> /cns/go-d/...). Override: TPU_STAGE_WS_ARCHIVE_METRO or
  TPU_STAGE_WS_ARCHIVE_ROOT. Unresolvable => --execute refuses.

RESTORE (`restore --xid X [--dry-run] [--into <google3 root>]`). Looks X up in
the local index (or --index <file or /cns path>), re-creates the deleted client
under its ORIGINAL name at its recorded baseline CL (`g4 citc -c <CL> <name>`)
unless --into names an existing google3 root, fetches the tarball (same fileutil
gate), checks its sha256, and untars it to the original relative path. Then
`tpu queue --resume_xid=X` resolves exactly as before: a relative registry
stagedir is found by the wrapper's workspace search, an absolute one by the
recreated client name.

DAILY CHECK. stage_ws_rotate.py `ensure` (called by tpu_wrapper.sh on every
non-resume stage) calls daily_check() at most once per 24 h: it counts the
operator's CitC clients, prints and logs a WARNING above 60
(TPU_STAGE_WS_CLIENT_WARN), and starts a detached `retire --daily` report that
logs the candidates to ~/.tpu_stage_ws_rotate.log. It never archives or
deletes. TPU_STAGE_WS_RETIRE_CHECK=0 disables it.

Exit codes: 0 ok (report, nothing to do, execute succeeded); 1 refused or
failed (nothing deleted for the failing client); 2 lock busy.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Iterable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
  sys.path.insert(0, _HERE)
import stage_ws_rotate as rot  # noqa: E402  (sibling module; shares config)

STAGES_REL = "experimental/qiaos/eqr_jax_final_stages"
_ARCHIVE_SUFFIX = "home/qiaos/eqr_data/stage_ws_retired"
_DEFAULT_ARCHIVE_METRO = "cmh"  # nearest group-storage metro to us-east4 (iad)
_TS_KEYS_RX = re.compile(r"(_at|_ts)$|^(ts|time|timestamp)$")
_XID_RX = re.compile(r"^\d{6,}$")
_STAGE_TOKEN_RX = re.compile(r"[^\s\"'=,;()\[\]{}<>]*eqr_jax_final_stages/[^\s\"'=,;()\[\]{}<>/]+")
# Terminal values (rule 4; the module docstring names each writer). Anything
# not listed -- SUBMITTED, RUNNING, QUEUED, a typo, a missing value -- is live.
QUEUE_TERMINAL = frozenset({"DONE", "FAILED"})  # == route_lib.FINISHED_STATES
REGISTRY_TERMINAL = frozenset({"COMPLETED", "CANCELLED", "FAILED",
                               "TERMINAL_RECONCILED", "TERMINAL_NOT_RUNNING"})
# Entry verdict classes, in report order.
V_ENDED_OLD = "ended-old"      # terminal and ended > ended_days ago: no block
V_LIVE = "live"                # live / unknown / missing status: blocks
V_NO_END = "no-end-ts"         # terminal, end time unknown: blocks
V_RECENT = "recent"            # terminal, ended <= ended_days ago: blocks


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
  try:
    return float(os.environ.get(name, "") or default)
  except ValueError:
    return default


def _paths(env: str, default: list) -> list:
  v = os.environ.get(env)
  if v is None:
    return default
  return [os.path.expanduser(p) for p in v.split(os.pathsep) if p]


def default_archive_root(metro: str) -> str:
  """<metro's group storage cell>/home/qiaos/eqr_data/stage_ws_retired, or ''.

  Resolved through cell_locality.py (the one measured cell -> metro -> bucket
  table); '' when the metro has no group storage or the table cannot be loaded,
  and --execute then refuses.
  """
  try:
    path = os.path.join(_HERE, "google3_tpu_utils", "cell_locality.py")
    spec = importlib.util.spec_from_file_location("_retire_cell_locality", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cells = mod.cells_in_metro(metro)
    if not cells:
      return ""
    return mod.bucket_for(cells[0], _ARCHIVE_SUFFIX, allow_personal=False)
  except Exception:  # pylint: disable=broad-except
    return ""


def cfg() -> dict:
  c = rot._cfg()  # pylint: disable=protected-access
  state_dir = os.path.dirname(c["log"])
  home = os.path.expanduser("~")
  h = lambda p: os.path.join(home, p)  # noqa: E731
  metro = os.environ.get("TPU_STAGE_WS_ARCHIVE_METRO", _DEFAULT_ARCHIVE_METRO)
  c.update({
      "state_dir": state_dir,
      "min_manifest": rot._env_int("TPU_STAGE_WS_RETIRE_MIN_MANIFEST", 9437184),  # pylint: disable=protected-access
      "idle_days": _env_float("TPU_STAGE_WS_RETIRE_IDLE_DAYS", 30.0),
      "record_days": _env_float("TPU_STAGE_WS_RETIRE_RECORD_DAYS", 30.0),
      "ended_days": _env_float("TPU_STAGE_WS_RETIRE_ENDED_DAYS", 30.0),
      "client_warn": rot._env_int("TPU_STAGE_WS_CLIENT_WARN", 60),  # pylint: disable=protected-access
      "queues": _paths("TPU_STAGE_WS_RETIRE_QUEUES", [
          h(".tpu_local_queue.json"), h("lyy-work/.npu_local_queue.json")]),
      "registries": _paths("TPU_STAGE_WS_RETIRE_REGISTRIES", [
          h(".tpu_jobs.json"), h("lyy-work/.npu_jobs.json")]),
      "aged": _paths("TPU_STAGE_WS_RETIRE_AGED", [
          h(".tpu_jobs_legacy.json"), h("lyy-work/.npu_jobs_legacy.json"),
          h(".tpu_ca_provenance.jsonl"), h(".tpu_jobs_hrm_trm_paper.json"),
          h(".tpu_jobs_v2.json"), h(".tpu_jobs_v2_archive.json")]),
      "archive_metro": metro,
      "archive_root": (os.environ.get("TPU_STAGE_WS_ARCHIVE_ROOT")
                       or default_archive_root(metro)),
      "fileutil": os.environ.get("TPU_STAGE_WS_FILEUTIL", "fileutil"),
      "pgrep": os.environ.get("TPU_STAGE_WS_PGREP", "pgrep"),
      "tar": os.environ.get("TPU_STAGE_WS_TAR", "tar"),
      "fileutil_max": 12,
      "fileutil_wait_s": _env_float("TPU_STAGE_WS_FILEUTIL_WAIT_S", 30.0),
      "fileutil_wait_max_s": _env_float("TPU_STAGE_WS_FILEUTIL_WAIT_MAX_S", 3600.0),
      "tmp": os.environ.get("TPU_STAGE_WS_RETIRE_TMP", h("tmp/stage_ws_retire")),
      "index": os.path.join(state_dir, ".tpu_stage_ws_retired_index.jsonl"),
      "stage_lock": os.environ.get("TPU_STAGE_WS_STAGE_LOCK",
                                   f"/tmp/tpu_stage.{os.getuid()}.lock"),
      "stage_lock_wait_s": _env_float("TPU_STAGE_WS_STAGE_LOCK_WAIT_S", 900.0),
      "stamp": os.path.join(state_dir, ".tpu_stage_ws_retire.stamp"),
      "retire_lock": os.path.join(state_dir, ".tpu_stage_ws_retire.lock"),
  })
  return c


def _now_ts() -> float:
  return time.time()


def _iso(ts: Optional[float]) -> str:
  if ts is None:
    return "unknown"
  return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime(
      "%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(v) -> Optional[float]:
  """Epoch seconds from an ISO string (naive = UTC) or an epoch number."""
  if isinstance(v, bool):
    return None
  if isinstance(v, (int, float)):
    return float(v) if 1e9 < v < 1e11 else None
  if isinstance(v, str) and len(v) >= 10 and v[:4].isdigit():
    s = v.strip().replace("Z", "+00:00")
    try:
      d = _dt.datetime.fromisoformat(s)
    except ValueError:
      try:
        d = _dt.datetime.strptime(v[:19], "%Y-%m-%dT%H:%M:%S")
      except ValueError:
        return None
    if d.tzinfo is None:
      d = d.replace(tzinfo=_dt.timezone.utc)
    return d.timestamp()
  return None


# --------------------------------------------------------------------------
# references from records
# --------------------------------------------------------------------------
class Ref:
  """One record naming one stagedir."""

  __slots__ = ("base", "client", "kind", "source", "xid", "ts", "blocking",
               "note")

  def __init__(self, base, client, kind, source, xid, ts, blocking, note=""):
    self.base = base          # stagedir basename
    self.client = client      # client name, None (relative: any client), or
                              # "<elsewhere>" (absolute path outside citc_root)
    self.kind = kind          # queue | active | aged | provenance
    self.source = source      # file path
    self.xid = xid
    self.ts = ts              # record / end time (epoch) or None
    self.blocking = blocking
    self.note = note          # queue/active: the entry's status and end time

  def describe(self) -> str:
    who = ("record" if not self.xid else self.xid if self.xid.startswith("(")
           else f"XID {self.xid}")
    if self.kind in ("queue", "active"):
      tail = f" [{self.note}]" if self.note else ""
    else:
      tail = f" @{_iso(self.ts)}"
    return f"{self.kind}:{os.path.basename(self.source)} {who}{tail}"


def _stage_paths(obj) -> Iterable[str]:
  """Every string token in a JSON value that names a stagedir."""
  stack = [obj]
  while stack:
    o = stack.pop()
    if isinstance(o, dict):
      for k, v in o.items():
        if k in ("stagedir", "abs_stagedir") and isinstance(v, str) and v \
            and "eqr_jax_final_stages/" not in v:
          yield v.rstrip("/")
        stack.append(v)
    elif isinstance(o, list):
      stack.extend(o)
    elif isinstance(o, str) and "eqr_jax_final_stages/" in o:
      for m in _STAGE_TOKEN_RX.finditer(o):
        yield m.group(0)


def _split_stage_path(citc_root: str, p: str):
  """(basename, client) for a stagedir path; client None when relative."""
  tail = p.split("eqr_jax_final_stages/", 1)[-1] if "eqr_jax_final_stages/" in p \
      else os.path.basename(p.rstrip("/"))
  base = tail.split("/", 1)[0]
  if not base:
    return None, None
  if p.startswith("/"):
    pre = citc_root.rstrip("/") + "/"
    if p.startswith(pre):
      return base, p[len(pre):].split("/", 1)[0]
    return base, "<elsewhere>"
  return base, None


def _record_refs(citc_root: str, rec) -> list:
  """[(base, client)] named by one record; an absolute mention of a basename
  supersedes a relative mention of it in the same record (provenance stores
  both `dir` and `rel` for one stage)."""
  pairs = set()
  for p in _stage_paths(rec):
    b, c = _split_stage_path(citc_root, p)
    if b:
      pairs.add((b, c))
  abs_bases = {b for b, c in pairs if c is not None}
  return sorted((b, c) for b, c in pairs if c is not None or b not in abs_bases)


def _record_ts(rec) -> Optional[float]:
  best = None
  if isinstance(rec, dict):
    for k, v in rec.items():
      if _TS_KEYS_RX.search(k):
        t = _parse_ts(v)
        if t is not None and (best is None or t > best):
          best = t
  return best


def _xids_in(obj) -> set:
  out = set()
  stack = [obj]
  while stack:
    o = stack.pop()
    if isinstance(o, dict):
      for k, v in o.items():
        if k in ("xid", "xids", "prior_xids", "resume_xid"):
          for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, (str, int)) and _XID_RX.match(str(x)):
              out.add(str(x))
        if isinstance(v, (dict, list)):
          stack.append(v)
    elif isinstance(o, list):
      stack.extend(o)
  return out


def _all_ts(obj) -> list:
  """Every timestamp under a *_at / ts key anywhere in a JSON value."""
  out = []
  stack = [obj]
  while stack:
    o = stack.pop()
    if isinstance(o, dict):
      for k, v in o.items():
        if isinstance(v, (dict, list)):
          stack.append(v)
        elif _TS_KEYS_RX.search(str(k)):
          t = _parse_ts(v)
          if t is not None:
            out.append(t)
    elif isinstance(o, list):
      stack.extend(o)
  return out


def _queue_end(row: dict) -> Optional[float]:
  """End time of a queue entry: its newest *_at, but only when its final
  attempt (highest submissions[].seq) recorded `ended_at`; else None."""
  subs = row.get("submissions")
  subs = [s for s in subs if isinstance(s, dict)] if isinstance(subs, list) else []
  if not subs:
    return None
  last = max(enumerate(subs), key=lambda p: (
      p[1]["seq"] if isinstance(p[1].get("seq"), int) else -1, p[0]))[1]
  if _parse_ts(last.get("ended_at")) is None:
    return None
  return max(_all_ts(row))


def _load_journal(reg_path: str, r: "Records") -> dict:
  """<registry>.journal(.1) -> {xid: [newest line ts, newest status-change ts,
  the status that change set]}. Unreadable => r.errors (plan INCOMPLETE)."""
  out = {}
  for jp in (reg_path + ".journal.1", reg_path + ".journal"):
    if not os.path.exists(jp):
      continue
    try:
      with open(jp, encoding="utf-8", errors="replace") as f:
        for ln in f:
          try:
            rec = json.loads(ln)
          except ValueError:
            continue  # a torn line; the status check below stays strict
          if not isinstance(rec, dict):
            continue
          t = _parse_ts(rec.get("ts"))
          if t is None:
            continue
          ch = rec.get("changed") if isinstance(rec.get("changed"), dict) else {}
          touched = list(ch)
          for k in ("added", "removed"):
            if isinstance(rec.get(k), list):
              touched.extend(rec[k])
          for x in touched:
            e = out.setdefault(str(x), [None, None, None])
            if e[0] is None or t >= e[0]:
              e[0] = t
          for x, fields in ch.items():
            st = fields.get("status") if isinstance(fields, dict) else None
            if isinstance(st, list) and len(st) == 2:
              e = out[str(x)]
              if e[1] is None or t >= e[1]:
                e[1], e[2] = t, st[1]
    except OSError as e:
      r.errors.append(f"{jp}: {e!r}")
  return out


def _registry_end(xid: str, rec: dict, jrnl: dict, qend: dict) -> Optional[float]:
  """End / last-update time of an active-registry row (rule 4), or None."""
  cands = [t for t in (_record_ts(rec), qend.get(xid)) if t is not None]
  j = jrnl.get(xid)
  if j and j[2] is not None and j[2] == rec.get("status"):
    cands.append(j[0])  # the journal saw the move to the current status
  return max(cands) if cands else None


def _verdict(status, terminal: frozenset, end: Optional[float], now: float,
             ended_s: float) -> str:
  if not isinstance(status, str) or status not in terminal:
    return V_LIVE
  if end is None:
    return V_NO_END
  if now - end <= ended_s:
    return V_RECENT
  return V_ENDED_OLD


def _entry_note(status, verdict: str, end: Optional[float]) -> str:
  st = status if isinstance(status, str) and status else "no status"
  if verdict == V_LIVE:
    return st
  if verdict == V_NO_END:
    return f"{st}, no end timestamp"
  return f"{st}, ended {_iso(end)}"


def _counts_note(counts: dict, days: float) -> str:
  blk = counts.get(V_LIVE, 0) + counts.get(V_NO_END, 0) + counts.get(V_RECENT, 0)
  return (f"terminal & ended >{days:g}d: {counts.get(V_ENDED_OLD, 0)} (no "
          f"block); blocking: {blk} = {counts.get(V_LIVE, 0)} live/unknown "
          f"status + {counts.get(V_NO_END, 0)} terminal without end timestamp "
          f"+ {counts.get(V_RECENT, 0)} terminal ended <={days:g}d")


class Records:
  """Every stagedir reference, by basename, plus per-client provenance times."""

  def __init__(self):
    self.by_base = {}       # base -> [Ref]
    self.prov_last = {}     # client -> newest provenance ts
    self.errors = []        # unreadable sources => plan INCOMPLETE
    self.sources = []       # (path, kind, n_records, n_refs, note)
    self.entry_counts = {}  # queue/active path -> {verdict class: n entries}
    self.unresolved_queue_xids = 0

  def add(self, ref: Ref) -> None:
    self.by_base.setdefault(ref.base, []).append(ref)


def _load_json(path: str):
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def _registry_items(d):
  if isinstance(d, dict):
    for k, v in d.items():
      if isinstance(v, dict):
        yield str(k), v
  elif isinstance(d, list):
    for v in d:
      if isinstance(v, dict):
        yield str(v.get("xid") or ""), v


def load_records(c: dict, now: Optional[float] = None) -> Records:
  now = _now_ts() if now is None else now
  young_s = c["record_days"] * 86400
  ended_s = c["ended_days"] * 86400
  root = c["citc_root"]
  r = Records()
  xid_map = {}  # xid -> {(base, client)} across every registry

  def _missing(path, kind):
    r.sources.append((path, kind, 0, 0, "absent"))

  # Pass 1: read every JSON registry and queue; build xid -> stagedirs (all
  # registries) and xid -> queue-recorded end time, which pass 2 needs.
  regs = []    # (kind, path, data, file mtime)
  for kind, paths in (("active", c["registries"]), ("aged", c["aged"])):
    for path in paths:
      if path.endswith(".jsonl"):
        continue
      if not os.path.exists(path):
        _missing(path, kind)
        continue
      try:
        d = _load_json(path)
        fmt = os.stat(path).st_mtime
      except (OSError, ValueError) as e:
        r.errors.append(f"{path}: {e!r}")
        continue
      regs.append((kind, path, d, fmt))
      for xid, rec in _registry_items(d):
        pairs = _record_refs(root, rec)
        if xid and pairs:
          xid_map.setdefault(xid, set()).update(pairs)
  queues = []  # (path, rows)
  for path in c["queues"]:
    if not os.path.exists(path):
      queues.append((path, None))
      continue
    try:
      d = _load_json(path)
    except (OSError, ValueError) as e:
      r.errors.append(f"{path}: {e!r}")
      continue
    rows = d.get("entries", d) if isinstance(d, dict) else d
    if isinstance(rows, dict):
      rows = list(rows.values())
    if not isinstance(rows, list):
      r.errors.append(f"{path}: unexpected queue shape {type(rows).__name__}")
      continue
    queues.append((path, [x for x in rows if isinstance(x, dict)]))
  qend = {}  # xid -> newest submissions[].ended_at the queues recorded for it
  for _, rows in queues:
    for row in rows or ():
      subs = row.get("submissions")
      for s in subs if isinstance(subs, list) else ():
        if isinstance(s, dict) and s.get("xid") is not None:
          t = _parse_ts(s.get("ended_at"))
          x = str(s["xid"])
          if t is not None and t > qend.get(x, 0):
            qend[x] = t

  def _emit(kind, path, row, own_pairs, own_xid, verdict, end, label):
    """Refs for one queue/active entry: the stagedirs it names, plus those of
    every XID it lists (so a live resume keeps its old terminal XIDs' stagedirs
    blocked). label: '(entry <job_id>)' for a queue entry, None for a row."""
    blocking = verdict != V_ENDED_OLD
    note = _entry_note(row.get("state" if kind == "queue" else "status"),
                       verdict, end)
    pairs = {(b, cl, own_xid) for b, cl in own_pairs}
    for x in _xids_in(row):
      if x == own_xid:
        continue
      if x in xid_map:
        pairs.update((b, cl, x) for b, cl in xid_map[x])
      elif kind == "queue":
        r.unresolved_queue_xids += 1
    for b, cl, x in sorted(pairs, key=str):
      r.add(Ref(b, cl, kind, path, x or label, end, blocking,
                note if label is None or x is None
                else f"{note}; entry {label[7:-1]}"))
    return len(pairs)

  # Pass 2a: registries. Active: an entry blocks unless terminal AND ended
  # more than ended_days ago. Aged: block when younger than record_days or of
  # unknown age.
  for kind, path, d, fmt in regs:
    jrnl = _load_journal(path, r) if kind == "active" else {}
    counts = {}
    n = nrefs = 0
    for xid, rec in _registry_items(d):
      n += 1
      if kind == "active":
        end = _registry_end(xid, rec, jrnl, qend)
        v = _verdict(rec.get("status"), REGISTRY_TERMINAL, end, now, ended_s)
        counts[v] = counts.get(v, 0) + 1
        nrefs += _emit(kind, path, rec, _record_refs(root, rec), xid or None,
                       v, end, None)
        continue
      pairs = _record_refs(root, rec)
      if not pairs:
        continue
      ts = _record_ts(rec)
      if ts is None and now - fmt > young_s:
        ts = fmt  # the file is older than the cut, so every record in it is
      blocking = ts is None or now - ts < young_s
      for b, cl in pairs:
        r.add(Ref(b, cl, kind, path, xid or None, ts, blocking))
        nrefs += 1
    if kind == "active":
      r.entry_counts[path] = counts
    r.sources.append((path, kind, n, nrefs,
                      _counts_note(counts, c["ended_days"])
                      if kind == "active" else ""))

  # Pass 2b: provenance (JSON lines): per-stage records with an absolute `dir`.
  for path in [p for p in c["aged"] if p.endswith(".jsonl")]:
    if not os.path.exists(path):
      _missing(path, "provenance")
      continue
    n = nrefs = bad = 0
    try:
      with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
          ln = ln.strip()
          if not ln:
            continue
          n += 1
          try:
            rec = json.loads(ln)
            ts = _record_ts(rec)
          except ValueError:
            bad += 1  # a torn line: keep whatever it names, as young
            rec, ts = ln, None
          blocking = ts is None or now - ts < young_s
          for b, cl in _record_refs(root, rec):
            r.add(Ref(b, cl, "provenance", path, None, ts, blocking))
            nrefs += 1
            if cl and cl != "<elsewhere>" and ts is not None:
              if ts > r.prov_last.get(cl, 0):
                r.prov_last[cl] = ts
    except OSError as e:
      r.errors.append(f"{path}: {e!r}")
      continue
    r.sources.append((path, "provenance", n, nrefs,
                      f"{bad} unparsable line(s) kept as young" if bad else ""))

  # Pass 2c: local queues, same rule as the active registries, directly and
  # through every XID an entry lists.
  for path, rows in queues:
    if rows is None:
      _missing(path, "queue")
      continue
    counts = {}
    nrefs = 0
    for row in rows:
      end = _queue_end(row)
      v = _verdict(row.get("state"), QUEUE_TERMINAL, end, now, ended_s)
      counts[v] = counts.get(v, 0) + 1
      label = f"(entry {row.get('job_id') or row.get('name') or '?'})"
      nrefs += _emit("queue", path, row, _record_refs(root, row), None, v, end,
                     label)
    r.entry_counts[path] = counts
    r.sources.append((path, "queue", len(rows), nrefs,
                      _counts_note(counts, c["ended_days"])))
  return r


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
def list_clients(c: dict) -> list:
  try:
    return sorted(n for n in os.listdir(c["citc_root"]) if not n.startswith("."))
  except OSError:
    return []


def _family_of(name: str) -> Optional[str]:
  for fam, prefix in rot.FAMILIES.items():
    if rot._name_re(prefix).match(name):  # pylint: disable=protected-access
      return fam
  return None


def _scan_stagedirs(c: dict, name: str):
  """({basename: mtime}, parent mtime) or (None, None) if unreadable."""
  parent = os.path.join(c["citc_root"], name, "google3", STAGES_REL)
  try:
    pm = os.stat(parent).st_mtime
  except FileNotFoundError:
    return {}, None
  except OSError:
    return None, None
  out = {}
  try:
    with os.scandir(parent) as it:
      for e in it:
        try:
          if e.is_dir(follow_symlinks=False):
            out[e.name] = e.stat(follow_symlinks=False).st_mtime
        except OSError:
          return None, None
  except OSError:
    return None, None
  return out, pm


def plan(c: dict, *, full: bool = True, now: Optional[float] = None,
         records: Optional[Records] = None) -> dict:
  """The retirement plan. Pure reads; never writes or deletes anything.

  full=False (the daily check) skips the stagedir scan and the record load for
  members that already fail the cheap rules 1-2.
  """
  now = _now_ts() if now is None else now
  clients = list_clients(c)
  out = {"now": now, "citc_root": c["citc_root"], "clients": clients,
         "families": {}, "manual": [], "members": [], "complete": True,
         "errors": [], "records": records}
  for n in clients:
    if _family_of(n) is None:
      out["manual"].append(n)
  bad = rot._bad_names(c)  # pylint: disable=protected-access
  for fam in sorted(rot.FAMILIES):
    infos = [rot.inspect(c, n, bad) for n in rot.members(c, fam)]
    healthy = [i for i in infos if i["healthy"]]
    protected = (healthy[-1] if healthy else (infos[-1] if infos else None))
    out["families"][fam] = [i["name"] for i in infos]
    for i in infos:
      m = {"name": i["name"], "family": fam, "manifest": i["manifest"],
           "healthy": i["healthy"], "cl": i["cl"], "torch": i["torch"],
           "reasons": [], "stagedirs": None, "last_stage": None,
           "archive": {}, "blocking": {}, "unreferenced": 0,
           "retirable": False}
      if protected is not None and i["name"] == protected["name"]:
        m["reasons"].append("newest healthy member" if healthy
                            else "newest member (no member is healthy)")
      if not 0 <= i["manifest"]:
        m["reasons"].append("manifest size unknown")
      elif i["manifest"] < c["min_manifest"]:
        m["reasons"].append(f"manifest {i['manifest']} < {c['min_manifest']}"
                            f" (not rotated out)")
      out["members"].append(m)

  need = [m for m in out["members"] if full or not m["reasons"]]
  if need and records is None:
    records = load_records(c, now)
    out["records"] = records
  if records is not None and records.errors:
    out["complete"] = False
    out["errors"] = list(records.errors)
  for m in need:
    sd, pm = _scan_stagedirs(c, m["name"])
    if sd is None:
      m["reasons"].append("staging parent unreadable")
      continue
    m["stagedirs"] = sd
    times = [t for t in [pm, *sd.values(), records.prov_last.get(m["name"])]
             if t is not None]
    m["last_stage"] = max(times) if times else None
    if m["last_stage"] is None:
      m["reasons"].append("last stage unknown")
    else:
      idle = (now - m["last_stage"]) / 86400
      if idle <= c["idle_days"]:
        m["reasons"].append(f"last stage {idle:.1f}d ago (<= {c['idle_days']:g}d)")
    for b in sorted(sd):
      refs = [x for x in records.by_base.get(b, ())
              if x.client is None or x.client == m["name"]]
      if not refs:
        m["unreferenced"] += 1
        continue
      blk = [x for x in refs if x.blocking]
      if blk:
        m["blocking"][b] = blk
      else:
        m["archive"][b] = refs
    if m["blocking"]:
      kinds = {}
      for refs in m["blocking"].values():
        for x in refs:
          kinds[x.kind] = kinds.get(x.kind, 0) + 1
      m["reasons"].append(
          f"{len(m['blocking'])} stagedir(s) referenced by live or recently "
          f"ended entries / young records ("
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ")")
  for m in out["members"]:
    m["retirable"] = (out["complete"] and not m["reasons"]
                      and m["stagedirs"] is not None)
  if not out["complete"]:
    for m in out["members"]:
      if m["stagedirs"] is not None and not m["reasons"]:
        m["reasons"].append("record sources incomplete (see errors)")
  return out


def format_report(c: dict, p: dict, *, verbose: bool = True) -> str:
  L = []
  n = len(p["clients"])
  L.append(f"stage-ws retire report (REPORT-ONLY, nothing archived or deleted) "
           f"at {_iso(p['now'])}")
  warn = " -- WARNING: above the limit" if n > c["client_warn"] else ""
  L.append(f"CitC clients under {p['citc_root']}: {n} (warn above "
           f"{c['client_warn']}){warn}")
  L.append(f"rules: not the newest healthy member | manifest >= "
           f"{c['min_manifest']} | last stage > {c['idle_days']:g}d ago | no "
           f"stagedir referenced by a queue / active-registry entry that is "
           f"live, of unknown status, without an end timestamp or ended <= "
           f"{c['ended_days']:g}d ago (terminal = queue "
           f"{'/'.join(sorted(QUEUE_TERMINAL))}, registry "
           f"{'/'.join(sorted(REGISTRY_TERMINAL))}), nor by another record "
           f"younger than {c['record_days']:g}d")
  L.append(f"archive root (used only by --execute): "
           f"{c['archive_root'] or 'UNRESOLVED (execute would refuse)'}")
  rec = p.get("records")
  if rec is not None and verbose:
    L.append("record sources:")
    for path, kind, nr, nf, note in rec.sources:
      L.append(f"  {kind:<10} {path}: {nr} records, {nf} stagedir refs"
               f"{' (' + note + ')' if note else ''}")
    if rec.unresolved_queue_xids:
      L.append(f"  (queue XIDs with no registry stagedir: "
               f"{rec.unresolved_queue_xids})")
  elif rec is None:
    L.append("record sources: not loaded (no member passed rules 1-2)")
  if p["errors"]:
    L.append("INCOMPLETE -- nothing is retirable while these sources are "
             "unreadable:")
    L.extend(f"  {e}" for e in p["errors"])
  for fam in sorted(p["families"]):
    L.append(f"family {fam} ({rot.FAMILIES[fam]}*): "
             f"{len(p['families'][fam])} member(s)")
    for m in [x for x in p["members"] if x["family"] == fam]:
      sd = m["stagedirs"]
      sds = "?" if sd is None else str(len(sd))
      verdict = "RETIRABLE" if m["retirable"] else "keep: " + "; ".join(m["reasons"])
      L.append(f"  {m['name']:<34} manifest={m['manifest']:<9} "
               f"stagedirs={sds:<4} last_stage={_iso(m['last_stage'])}  {verdict}")
      if m["retirable"] or (verbose and m["blocking"]):
        if m["blocking"]:
          for b, refs in sorted(m["blocking"].items())[:3]:
            L.append(f"      blocked: {b} <- " +
                     ", ".join(x.describe() for x in refs[:3]))
          if len(m["blocking"]) > 3:
            L.append(f"      ... and {len(m['blocking']) - 3} more blocked "
                     f"stagedir(s)")
        if m["retirable"]:
          xids = sorted({x.xid for refs in m["archive"].values() for x in refs
                         if x.xid})
          L.append(f"      would archive {len(m['archive'])} referenced "
                   f"stagedir(s) ({len(xids)} XID(s)) to "
                   f"{c['archive_root'] or '?'}/{m['name']}/ ; "
                   f"{m['unreferenced']} unreferenced stagedir(s) would not "
                   f"be archived")
  ret = [m["name"] for m in p["members"] if m["retirable"]]
  L.append(f"RETIRABLE: {', '.join(ret) if ret else 'none'}")
  L.append(f"manual, not managed (never considered): {len(p['manual'])}: "
           + ", ".join(p["manual"]))
  return "\n".join(L)


# --------------------------------------------------------------------------
# fileutil (gated), tar, hashing
# --------------------------------------------------------------------------
def _log(c: dict, msg: str) -> None:
  rot.log(c, f"retire: {msg}")


def _file_log(c: dict, msg: str) -> None:
  """Append to the rotate log without echoing to stderr."""
  try:
    with open(c["log"], "a", encoding="utf-8") as f:
      f.write(f"{rot._now()} pid={os.getpid()} {msg}\n")  # pylint: disable=protected-access
  except OSError:
    pass


def _pgrep_count(c: dict) -> Optional[int]:
  try:
    p = subprocess.run(shlex.split(c["pgrep"]) + ["-c", "fileutil"],
                       capture_output=True, text=True, timeout=30)
  except (OSError, subprocess.TimeoutExpired):
    return None
  if p.returncode not in (0, 1):
    return None
  try:
    return int((p.stdout or "0").strip() or 0)
  except ValueError:
    return None


def fileutil_gate(c: dict) -> bool:
  """Wait (BEFORE a fileutil call) while more than 12 fileutil processes run.

  False when pgrep cannot answer or the wait exceeds fileutil_wait_max_s; the
  caller must then not call fileutil.
  """
  waited = 0.0
  while True:
    n = _pgrep_count(c)
    if n is None:
      _log(c, "pgrep -c fileutil failed; not calling fileutil")
      return False
    if n <= c["fileutil_max"]:
      return True
    if waited >= c["fileutil_wait_max_s"]:
      _log(c, f"{n} fileutil processes after {waited:.0f}s of waiting; giving up")
      return False
    _log(c, f"{n} fileutil processes running (> {c['fileutil_max']}); "
         f"sleeping {c['fileutil_wait_s']:g}s before the next fileutil call")
    time.sleep(c["fileutil_wait_s"])
    waited += c["fileutil_wait_s"]


def fileutil(c: dict, args: list, timeout: float = 1800) -> tuple:
  """(ok, detail). One gated, sequential fileutil call."""
  if not fileutil_gate(c):
    return False, "fileutil gate refused"
  cmd = shlex.split(c["fileutil"]) + list(args)
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=os.path.expanduser("~"))
  except (OSError, subprocess.TimeoutExpired) as e:
    return False, f"`{' '.join(cmd)}`: {e!r}"
  if p.returncode != 0:
    return False, (f"`{' '.join(cmd)}` rc={p.returncode}: "
                   f"{(p.stdout + p.stderr).strip()[-300:]}")
  return True, (p.stdout or "").strip()


def sha256_file(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _upload_verified(c: dict, local: str, dest: str, scratch: str) -> tuple:
  """Copy local -> dest, copy it back, require the same sha256."""
  want = sha256_file(local)
  ok, d = fileutil(c, ["cp", "-f", local, dest])
  if not ok:
    return False, f"upload failed: {d}"
  back = os.path.join(scratch, os.path.basename(local) + ".readback")
  ok, d = fileutil(c, ["cp", "-f", dest, back])
  if not ok:
    return False, f"read-back failed: {d}"
  try:
    got = sha256_file(back)
  finally:
    try:
      os.remove(back)
    except OSError:
      pass
  if got != want:
    return False, f"read-back sha256 {got[:12]} != {want[:12]} for {dest}"
  return True, want


# --------------------------------------------------------------------------
# execute: archive, then delete
# --------------------------------------------------------------------------
class _Flock:
  def __init__(self, path: str):
    self.path, self.fd = path, None

  def acquire(self, wait_s: float) -> bool:
    self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + max(0.0, wait_s)
    while True:
      try:
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
      except OSError as e:
        if e.errno not in (errno.EAGAIN, errno.EACCES):
          raise
        if time.time() >= deadline:
          os.close(self.fd)
          self.fd = None
          return False
        time.sleep(1.0)

  def release(self) -> None:
    if self.fd is not None:
      try:
        fcntl.flock(self.fd, fcntl.LOCK_UN)
      finally:
        os.close(self.fd)
        self.fd = None


def _index_rows(c: dict, m: dict, base: str, refs: list, tarball: str,
                sha: str, nbytes: int) -> list:
  xids = sorted({x.xid for x in refs if x.xid and _XID_RX.match(x.xid)})
  common = {
      "client": m["name"], "family": m["family"], "baseline_cl": m["cl"],
      "torch": m["torch"], "citc_root": c["citc_root"],
      "stagedir_rel": f"{STAGES_REL}/{base}", "basename": base,
      "tarball": tarball, "sha256": sha, "bytes": nbytes,
      "archived_at": _iso(_now_ts()),
      "records": sorted({x.describe() for x in refs})[:20],
  }
  return [dict(common, xid=x) for x in xids] or [dict(common, xid=None)]


def archive_client(c: dict, m: dict) -> tuple:
  """Steps 1-3 for one member. (ok, detail, index rows). Deletes nothing."""
  name = m["name"]
  root = c["archive_root"].rstrip("/")
  dest_dir = f"{root}/{name}"
  parent = os.path.join(c["citc_root"], name, "google3", STAGES_REL)
  scratch = os.path.join(c["tmp"], f"{name}.{int(_now_ts())}.{os.getpid()}")
  try:
    os.makedirs(scratch, exist_ok=False)
  except OSError as e:
    return False, f"scratch dir {scratch}: {e!r}", []
  with open(os.path.join(scratch, "MANIFEST.json"), "w", encoding="utf-8") as f:
    json.dump({"client": name, "archive": sorted(m["archive"]),
               "dest": dest_dir, "planned_at": _iso(_now_ts())}, f, indent=1)
  ok, d = fileutil(c, ["mkdir", "-p", dest_dir])
  if not ok:
    return False, f"mkdir {dest_dir}: {d}", []
  rows = []
  for base, refs in sorted(m["archive"].items()):
    tar_path = os.path.join(scratch, f"{base}.tar.gz")
    tcmd = shlex.split(c["tar"]) + ["-czf", tar_path, "-C", parent, base]
    try:
      p = subprocess.run(tcmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
      return False, f"tar {base}: {e!r}", []
    if p.returncode != 0 or not os.path.isfile(tar_path):
      return False, f"tar {base} rc={p.returncode}: {p.stderr.strip()[-300:]}", []
    try:
      p = subprocess.run(shlex.split(c["tar"]) + ["-tzf", tar_path],
                         capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
      return False, f"tar -t {base}: {e!r}", []
    first = (p.stdout.splitlines() or [""])[0]
    if p.returncode != 0 or not first.startswith(base):
      return False, (f"tar -t {base} rc={p.returncode} first={first!r}: "
                     f"the tarball does not list the stagedir"), []
    nbytes = os.path.getsize(tar_path)
    dest = f"{dest_dir}/{base}.tar.gz"
    ok, sha = _upload_verified(c, tar_path, dest, scratch)
    if not ok:
      return False, f"{base}: {sha}", []
    rows.extend(_index_rows(c, m, base, refs, dest, sha, nbytes))
    _log(c, f"{name}: archived {base} ({nbytes} B, sha256 {sha[:12]}) -> {dest}")
  idx = os.path.join(scratch, "INDEX.jsonl")
  with open(idx, "w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")
  ok, d = _upload_verified(c, idx, f"{dest_dir}/INDEX.jsonl", scratch)
  if not ok:
    return False, f"INDEX.jsonl: {d}", []
  try:
    with open(c["index"], "a", encoding="utf-8") as f:
      fcntl.flock(f.fileno(), fcntl.LOCK_EX)
      for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
      f.flush()
      os.fsync(f.fileno())
  except OSError as e:
    return False, f"local index {c['index']}: {e!r}", []
  return True, f"{len(m['archive'])} stagedir(s), {len(rows)} index row(s)", rows


def delete_client(c: dict, m: dict) -> tuple:
  """Step 4: under the wrapper's stage lock, re-plan; delete only if unchanged."""
  name = m["name"]
  lock = _Flock(c["stage_lock"])
  if not lock.acquire(c["stage_lock_wait_s"]):
    return False, f"stage lock {c['stage_lock']} busy; not deleting {name}"
  try:
    fresh = plan(c, full=True)
    fm = next((x for x in fresh["members"] if x["name"] == name), None)
    if fm is None or not fm["retirable"]:
      why = "gone" if fm is None else "; ".join(fm["reasons"])
      return False, f"{name} is no longer retirable ({why}); not deleting"
    if set(fm["stagedirs"]) != set(m["stagedirs"]) \
        or set(fm["archive"]) != set(m["archive"]):
      return False, f"{name}: stagedirs changed since archiving; not deleting"
    cmd = shlex.split(c["g4"]) + ["citc", "-d", name]
    try:
      p = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                         cwd=os.path.expanduser("~"))
    except (OSError, subprocess.TimeoutExpired) as e:
      return False, f"`{' '.join(cmd)}`: {e!r}"
    out = (p.stdout + p.stderr).strip().replace("\n", " | ")[-300:]
    if p.returncode != 0:
      return False, f"`{' '.join(cmd)}` rc={p.returncode}: {out}"
    return True, f"deleted ({out})"
  finally:
    lock.release()


def execute(c: dict, p: dict, only: Optional[list] = None, out=print) -> int:
  if not p["complete"]:
    out("REFUSED: record sources incomplete: " + "; ".join(p["errors"]))
    return 1
  targets = [m for m in p["members"] if m["retirable"]
             and (not only or m["name"] in only)]
  if only:
    unknown = set(only) - {m["name"] for m in p["members"]}
    if unknown:
      out(f"REFUSED: not managed family members: {sorted(unknown)}")
      return 1
  if not targets:
    out("execute: nothing retirable; nothing done")
    return 0
  root = c["archive_root"]
  if not root.startswith("/cns/"):
    out(f"REFUSED: archive root unresolved or not on CNS ({root!r})")
    return 1
  for tool in (c["fileutil"], c["tar"], c["g4"], c["pgrep"]):
    if not shutil.which(shlex.split(tool)[0]):
      out(f"REFUSED: {tool!r} not found")
      return 1
  for m in targets:
    _log(c, f"EXECUTE {m['name']}: archiving {len(m['archive'])} stagedir(s)")
    ok, detail, _ = archive_client(c, m)
    if not ok:
      _log(c, f"EXECUTE {m['name']}: archive FAILED ({detail}); NOT deleting; stop")
      out(f"REFUSED: {m['name']}: archive step failed: {detail}; nothing deleted")
      return 1
    _log(c, f"EXECUTE {m['name']}: archived ({detail}); deleting")
    ok, detail = delete_client(c, m)
    _log(c, f"EXECUTE {m['name']}: {'deleted' if ok else 'NOT deleted'}: {detail}")
    out(f"{m['name']}: {'DELETED' if ok else 'NOT deleted'}: {detail}")
    if not ok:
      return 1
  return 0


# --------------------------------------------------------------------------
# restore
# --------------------------------------------------------------------------
def _load_index(c: dict, index: Optional[str], scratch: str) -> list:
  path = index or c["index"]
  if path.startswith("/cns/"):
    local = os.path.join(scratch, "INDEX.jsonl")
    ok, d = fileutil(c, ["cp", "-f", path, local])
    if not ok:
      raise RuntimeError(f"fetch {path}: {d}")
    path = local
  rows = []
  with open(path, encoding="utf-8") as f:
    for ln in f:
      ln = ln.strip()
      if ln:
        rows.append(json.loads(ln))
  return rows


def restore(c: dict, *, xid: Optional[str] = None, stagedir: Optional[str] = None,
            into: Optional[str] = None, index: Optional[str] = None,
            dry_run: bool = False, out=print) -> int:
  scratch = os.path.join(c["tmp"], f"restore.{int(_now_ts())}.{os.getpid()}")
  os.makedirs(scratch, exist_ok=True)
  try:
    rows = _load_index(c, index, scratch)
  except (OSError, ValueError, RuntimeError) as e:
    out(f"restore: cannot read the index: {e}")
    return 1
  want_base = os.path.basename(stagedir.rstrip("/")) if stagedir else None
  hits = [r for r in rows if (xid and str(r.get("xid")) == str(xid))
          or (want_base and r.get("basename") == want_base)]
  if not hits:
    out(f"restore: no index row for {'XID ' + str(xid) if xid else want_base}")
    return 1
  row = hits[-1]
  client, base = row["client"], row["basename"]
  ws = os.path.join(row.get("citc_root") or c["citc_root"], client)
  g3 = into or os.path.join(ws, "google3")
  dest_parent = os.path.join(g3, STAGES_REL)
  target = os.path.join(dest_parent, base)
  out(f"restore: XID {row.get('xid')} -> {row['tarball']} (sha256 "
      f"{row['sha256'][:12]}) -> {target}")
  if os.path.isdir(target):
    out("restore: already present; nothing to do")
    return 0
  steps = []
  if not into and not os.path.isdir(ws):
    steps.append(shlex.split(c["g4"]) + ["citc", "-c", str(row["baseline_cl"]),
                                         client])
  if dry_run:
    for s in steps:
      out(f"  would run: {' '.join(s)}")
    out(f"  would fetch {row['tarball']}, check sha256, untar into {dest_parent}")
    return 0
  for s in steps:
    try:
      p = subprocess.run(s, capture_output=True, text=True, timeout=600,
                         cwd=os.path.expanduser("~"))
    except (OSError, subprocess.TimeoutExpired) as e:
      out(f"restore: `{' '.join(s)}`: {e!r}")
      return 1
    if p.returncode != 0:
      out(f"restore: `{' '.join(s)}` rc={p.returncode}: "
          f"{(p.stdout + p.stderr).strip()[-300:]}")
      return 1
    if rot.baseline_cl(ws) != str(row["baseline_cl"]):
      out(f"restore: recreated {client} reads back CL {rot.baseline_cl(ws)!r}, "
          f"expected {row['baseline_cl']}; stopping")
      return 1
  local = os.path.join(scratch, f"{base}.tar.gz")
  ok, d = fileutil(c, ["cp", "-f", row["tarball"], local])
  if not ok:
    out(f"restore: fetch failed: {d}")
    return 1
  if sha256_file(local) != row["sha256"]:
    out("restore: sha256 mismatch on the fetched tarball; not extracting")
    return 1
  os.makedirs(dest_parent, exist_ok=True)
  p = subprocess.run(shlex.split(c["tar"]) + ["-xzf", local, "-C", dest_parent],
                     capture_output=True, text=True, timeout=3600)
  if p.returncode != 0 or not os.path.isdir(target):
    out(f"restore: untar rc={p.returncode}: {p.stderr.strip()[-300:]}")
    return 1
  os.remove(local)
  out(f"restore: OK. `tpu queue --resume_xid={row.get('xid')} ...` now finds "
      f"{target}" + ("" if not into else
                     " (relative registry stagedirs only: an ABSOLUTE record "
                     "still names the deleted client)"))
  return 0


# --------------------------------------------------------------------------
# daily check (called from stage_ws_rotate.ensure)
# --------------------------------------------------------------------------
def daily_check(c: Optional[dict] = None) -> str:
  """At most once per 24 h: count clients (WARNING above the limit) and start a
  detached report-only `retire --daily`. Never raises; never deletes."""
  try:
    c = c or cfg()
    if os.environ.get("TPU_STAGE_WS_RETIRE_CHECK", "1") == "0":
      return "disabled"
    try:
      if _now_ts() - os.stat(c["stamp"]).st_mtime < 86400:
        return "skip"
    except OSError:
      pass
    lock = _Flock(c["retire_lock"])
    if not lock.acquire(0):
      return "busy"
    try:
      try:
        if _now_ts() - os.stat(c["stamp"]).st_mtime < 86400:
          return "skip"
      except OSError:
        pass
      tmp = f"{c['stamp']}.tmp.{os.getpid()}"
      with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"{_iso(_now_ts())} pid={os.getpid()}\n")
      os.replace(tmp, c["stamp"])
      n = len(list_clients(c))
      if n > c["client_warn"]:
        msg = (f"WARNING: {n} CitC clients under {c['citc_root']} (> "
               f"{c['client_warn']}). Review `stage_ws_rotate.py retire` "
               f"(report-only) and the manual clients it lists.")
        print(f"[stage-ws-rotate] {msg}", file=sys.stderr, flush=True)
        _file_log(c, f"retire-check: {msg}")
      else:
        _file_log(c, f"retire-check: {n} CitC clients under {c['citc_root']} "
                     f"(warn above {c['client_warn']}); report running "
                     f"detached")
      try:
        logfd = os.open(c["log"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
      except OSError:
        logfd = subprocess.DEVNULL
      try:
        subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, os.path.abspath(__file__), "retire", "--daily"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=logfd,
            cwd=os.path.expanduser("~"), start_new_session=True, close_fds=True)
      finally:
        if isinstance(logfd, int) and logfd >= 0:
          os.close(logfd)
      return "started"
    finally:
      lock.release()
  except Exception as e:  # pylint: disable=broad-except
    try:
      _file_log(c or rot._cfg(), f"retire-check: error {e!r} (ignored)")  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
      pass
    return "error"


def _daily_report(c: dict) -> int:
  signal.alarm(900)  # a hung CitC read must not leave this child around
  rot._DETACHED = True  # pylint: disable=protected-access
  p = plan(c, full=False)
  cands = [m for m in p["members"] if m["retirable"]]
  keep = "; ".join(f"{m['name']}: {', '.join(m['reasons'])}"
                   for m in p["members"] if not m["retirable"])
  n = len(p["clients"])
  rot.log(c, f"retire-check: clients={n} (warn>{c['client_warn']}"
             f"{' WARNING' if n > c['client_warn'] else ''}) "
             f"members={len(p['members'])} manual={len(p['manual'])} "
             f"complete={p['complete']} candidates="
             f"{[m['name'] for m in cands] or 'none'} [{keep or '-'}]")
  for m in cands:
    xids = sorted({x.xid for refs in m["archive"].values() for x in refs
                   if x.xid})
    rot.log(c, f"retire-check: CANDIDATE {m['name']} manifest={m['manifest']} "
               f"last_stage={_iso(m['last_stage'])} archive={len(m['archive'])}"
               f" stagedir(s) xids={len(xids)} unreferenced={m['unreferenced']}"
               f" (report only; run `stage_ws_rotate.py retire` to see the plan)")
  for e in p["errors"]:
    rot.log(c, f"retire-check: INCOMPLETE {e}")
  return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  if not argv or argv[0].startswith("-"):
    argv = ["retire"] + argv
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  sub = ap.add_subparsers(dest="cmd", required=True)
  r = sub.add_parser("retire", help="report (default) or --execute")
  r.add_argument("--execute", action="store_true",
                 help="ARCHIVE then DELETE every retirable client")
  r.add_argument("--client", action="append", default=[],
                 help="with --execute: only these clients")
  r.add_argument("--json", action="store_true")
  r.add_argument("--daily", action="store_true", help=argparse.SUPPRESS)
  s = sub.add_parser("restore", help="restore an archived stagedir")
  g = s.add_mutually_exclusive_group(required=True)
  g.add_argument("--xid")
  g.add_argument("--stagedir")
  s.add_argument("--into", help="an existing google3 root to restore into")
  s.add_argument("--index", help="index file (default: the local index; a "
                 "/cns/.../INDEX.jsonl is fetched)")
  s.add_argument("--dry-run", "--dry_run", action="store_true")
  a = ap.parse_args(argv)
  c = cfg()
  if a.cmd == "restore":
    return restore(c, xid=a.xid, stagedir=a.stagedir, into=a.into,
                   index=a.index, dry_run=a.dry_run)
  if a.daily:
    return _daily_report(c)
  if a.client and not a.execute:
    ap.error("--client only applies to --execute")
  if a.execute:
    lock = _Flock(c["retire_lock"])
    if not lock.acquire(0):
      print("retire: another retire is running", file=sys.stderr)
      return 2
    try:
      p = plan(c, full=True)
      print(format_report(c, p, verbose=False))
      return execute(c, p, only=a.client or None)
    finally:
      lock.release()
  p = plan(c, full=True)
  if a.json:
    def _ser(m):
      d = dict(m)
      d["archive"] = {b: [x.describe() for x in v] for b, v in m["archive"].items()}
      d["blocking"] = {b: [x.describe() for x in v] for b, v in m["blocking"].items()}
      d["stagedirs"] = None if m["stagedirs"] is None else len(m["stagedirs"])
      return d
    print(json.dumps({"clients": p["clients"], "manual": p["manual"],
                      "complete": p["complete"], "errors": p["errors"],
                      "members": [_ser(m) for m in p["members"]]}, indent=1))
  else:
    print(format_report(c, p))
  return 0


if __name__ == "__main__":
  sys.exit(main())
