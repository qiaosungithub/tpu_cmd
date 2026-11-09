#!/usr/bin/env python3
"""Keep a torch >= 2.15 CitC staging workspace with headroom in each operator's pool.

WHY. `tpu queue` stages every job into a CitC workspace (STAGE_WS_ROOT) and the
workspace's baseline CL decides which //third_party/py/torch the job links.
tpu_wrapper.sh rejects any workspace whose .citc/manifest is >= 10 MiB (CitC's
auto-keyframe cliff sits near 12 MB; the limit must NOT be raised), so every
staging workspace eventually fills up and falls out of the pool. When the last
torch >= 2.15 one fell out (2026-11-07) the rotation silently moved builds to a
torch 2.13 workspace, where masked GQA SDPA falls back to the MATH kernel and
doc-mask TinyLlama runs OOM. This script creates the NEXT workspace before that
happens.

CONTRACT (shared with tpu_wrapper.sh, which globs the same names):
  * A FAMILY is a name prefix under /google/src/cloud/qiaos/ (qiaos owns every
    staging client, including the ones used for lyy's builds):
        qiaos -> tpu_staging_t215_<YYYYMMDD>[_<n>]
        lyy   -> tpu_staging_lyy_t215_<YYYYMMDD>[_<n>]
  * Every member is created with `g4 citc -c <CL> <name>` at the SAME baseline
    CL (default 989229158 = torch 2.15.0a0, the pin the fleet built against
    11-02..11-07), then verified: baseline CL read back, torch METADATA >= the
    minimum, eqr_jax_final_stages created, a probe file written and read back
    after 6 s and 12 s, dropped_resources.ascii still at its 49-byte header.
  * A member that fails verification is appended to <state dir>/.tpu_stage_ws_bad
    and the wrapper never uses it. Nothing is ever deleted: retired workspaces
    keep their stagedirs so --resume_xid keeps resolving.
  * The wrapper prefers members oldest first, so one fills to the 10 MiB limit
    before the next takes over; this script creates a new member as soon as no
    member has headroom (< 9 MiB by default), i.e. one step ahead.

COMMANDS
  status [--family F|all] [--json]   what exists, sizes, torch, health
  ensure --family F [--dry-run] [--background] [--wait SECS]
      Idempotent. Exit 0 and print the preferred healthy member when some member
      has headroom. Otherwise (under an exclusive flock, re-scanning after the
      lock is taken so concurrent callers never create duplicates) create and
      verify the next member. --background returns at once and does the work in
      a detached child (the wrapper's pre-create hook); --wait is how long to
      wait for a lock held by another creator (0 = do not wait).
      Exit codes: 0 ok / created / dry-run, 1 failed or refused (cooldown,
      daily cap), 2 lock busy.

SAFETY RAILS: at most TPU_STAGE_WS_MAX_CREATES_PER_DAY (default 2) creations per
family per 24 h, and a 30 min cooldown after a failed creation, so a bug or a
broken CitC cannot spray workspaces. Every action is logged to
<state dir>/.tpu_stage_ws_rotate.log.

Knobs (env): TPU_STAGE_WS_BASELINE_CL, TPU_STAGE_WS_MIN_TORCH (2.15),
TPU_STAGE_WS_MAX_MANIFEST_BYTES (10485760, the wrapper's limit),
TPU_STAGE_WS_PRECREATE_BYTES (9437184), TPU_STAGE_WS_MAX_CREATES_PER_DAY (2),
TPU_STAGE_WS_FAIL_COOLDOWN_S (1800), TPU_STAGE_WS_ROTATE_DIR ($HOME; lock, log,
state, bad list), and for tests TPU_STAGE_WS_CITC_ROOT, TPU_STAGE_WS_G4,
TPU_STAGE_WS_PROBE_SLEEP_S.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import errno
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Optional

FAMILIES = {
    "qiaos": "tpu_staging_t215_",
    "lyy": "tpu_staging_lyy_t215_",
}
_DROPPED_HEADER_BYTES = 49  # an idle dropped_resources.ascii is the bare header
_DETACHED = False  # set in the background child: stderr IS the log file


def _env_int(name: str, default: int) -> int:
  try:
    return int(os.environ.get(name, "") or default)
  except ValueError:
    return default


def _cfg() -> dict:
  state_dir = os.environ.get("TPU_STAGE_WS_ROTATE_DIR") or os.path.expanduser("~")
  return {
      "citc_root": os.environ.get("TPU_STAGE_WS_CITC_ROOT",
                                  "/google/src/cloud/qiaos"),
      "cl": os.environ.get("TPU_STAGE_WS_BASELINE_CL", "989229158"),
      "min_torch": os.environ.get("TPU_STAGE_WS_MIN_TORCH", "2.15"),
      "max_manifest": _env_int("TPU_STAGE_WS_MAX_MANIFEST_BYTES", 10485760),
      "precreate": _env_int("TPU_STAGE_WS_PRECREATE_BYTES", 9437184),
      "max_per_day": _env_int("TPU_STAGE_WS_MAX_CREATES_PER_DAY", 2),
      "fail_cooldown": _env_int("TPU_STAGE_WS_FAIL_COOLDOWN_S", 1800),
      "g4": os.environ.get("TPU_STAGE_WS_G4", "g4"),
      "probe_sleep": float(os.environ.get("TPU_STAGE_WS_PROBE_SLEEP_S", "6")),
      "lock": os.path.join(state_dir, ".tpu_stage_ws_rotate.lock"),
      "log": os.path.join(state_dir, ".tpu_stage_ws_rotate.log"),
      "state": os.path.join(state_dir, ".tpu_stage_ws_rotate_state.json"),
      "bad": os.path.join(state_dir, ".tpu_stage_ws_bad"),
      "heartbeat": os.path.join(state_dir, ".tpu_stage_ws_rotate.heartbeat"),
  }


# --------------------------------------------------------------------------
# logging / state
# --------------------------------------------------------------------------
def _now() -> str:
  return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(cfg: dict, msg: str, *, to_file: bool = True) -> None:
  line = f"{_now()} pid={os.getpid()} {msg}"
  if not _DETACHED:
    print(f"[stage-ws-rotate] {msg}", file=sys.stderr, flush=True)
  if to_file:
    try:
      with open(cfg["log"], "a", encoding="utf-8") as f:
        f.write(line + "\n")
    except OSError:
      pass


def _load_state(cfg: dict) -> dict:
  try:
    with open(cfg["state"], encoding="utf-8") as f:
      d = json.load(f)
    return d if isinstance(d, dict) else {}
  except (OSError, ValueError):
    return {}


def _save_state(cfg: dict, state: dict) -> None:
  tmp = cfg["state"] + f".tmp.{os.getpid()}"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=1, sort_keys=True)
  os.replace(tmp, cfg["state"])


def _bad_names(cfg: dict) -> set:
  try:
    with open(cfg["bad"], encoding="utf-8") as f:
      return {ln.strip() for ln in f if ln.strip()}
  except OSError:
    return set()


# --------------------------------------------------------------------------
# inspection
# --------------------------------------------------------------------------
def torch_version(g3: str) -> str:
  """First `version: "<major>.<minor>..."` in torch METADATA ('' if unknown).

  Same regex as tpu_wrapper.sh::_tpu_ws_torch_version: some METADATA files list
  a commit hash under `version:` first, which must not match.
  """
  try:
    with open(os.path.join(g3, "third_party/py/torch/METADATA"),
              encoding="utf-8", errors="replace") as f:
      for ln in f:
        m = re.match(r'^\s*version:\s*"(\d+\.\d+[^"]*)"', ln)
        if m:
          return m.group(1)
  except OSError:
    pass
  return ""


def torch_ge(version: str, minimum: str) -> bool:
  a = re.match(r"^(\d+)\.(\d+)", version or "")
  b = re.match(r"^(\d+)\.(\d+)", minimum or "")
  if not a:
    return False
  if not b:
    return True
  return (int(a.group(1)), int(a.group(2))) >= (int(b.group(1)), int(b.group(2)))


def baseline_cl(ws: str) -> str:
  try:
    with open(os.path.join(ws, ".citc/srcfs_workspace.ascii"), "rb") as f:
      m = re.search(rb"change:\s*(\d+)", f.read())
    return m.group(1).decode() if m else ""
  except OSError:
    return ""


def _size(path: str) -> int:
  try:
    return os.stat(path).st_size
  except OSError:
    return -1


def _name_re(prefix: str):
  return re.compile(r"^" + re.escape(prefix) + r"(\d{8})(?:_(\d+))?$")


def members(cfg: dict, family: str) -> list:
  """Existing family members, oldest first (the wrapper's preference order)."""
  prefix = FAMILIES[family]
  rx = _name_re(prefix)
  try:
    names = os.listdir(cfg["citc_root"])
  except OSError:
    names = []
  out = []
  for n in names:
    m = rx.match(n)
    if m:
      out.append((m.group(1), int(m.group(2) or 1), n))
  out.sort()
  return [n for _, _, n in out]


def inspect(cfg: dict, name: str, bad: Optional[set] = None) -> dict:
  ws = os.path.join(cfg["citc_root"], name)
  g3 = os.path.join(ws, "google3")
  info = {
      "name": name,
      "google3": g3,
      "manifest": _size(os.path.join(ws, ".citc/manifest")),
      "dropped": _size(os.path.join(ws, ".citc/dropped_resources.ascii")),
      "cl": baseline_cl(ws),
      "torch": torch_version(g3),
      "bad": name in (bad if bad is not None else _bad_names(cfg)),
  }
  reasons = []
  if not (os.path.isdir(g3) and os.path.isdir(os.path.join(ws, ".citc"))):
    reasons.append("missing")
  if not 0 <= info["dropped"] <= _DROPPED_HEADER_BYTES:
    reasons.append(f"dropped={info['dropped']}")
  if not 0 <= info["manifest"] < cfg["max_manifest"]:
    reasons.append(f"manifest={info['manifest']}>=limit")
  if not torch_ge(info["torch"], cfg["min_torch"]):
    reasons.append(f"torch={info['torch'] or 'UNKNOWN'}<{cfg['min_torch']}")
  if info["bad"]:
    reasons.append("listed-bad")
  info["healthy"] = not reasons
  info["headroom"] = info["healthy"] and info["manifest"] < cfg["precreate"]
  info["reasons"] = reasons
  return info


def scan(cfg: dict, family: str) -> list:
  bad = _bad_names(cfg)
  return [inspect(cfg, n, bad) for n in members(cfg, family)]


def next_name(cfg: dict, family: str, today: Optional[str] = None) -> str:
  prefix = FAMILIES[family]
  today = today or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
  base = prefix + today
  if not os.path.exists(os.path.join(cfg["citc_root"], base)):
    return base
  n = 2
  while os.path.exists(os.path.join(cfg["citc_root"], f"{base}_{n}")):
    n += 1
  return f"{base}_{n}"


def _preferred(infos: list) -> Optional[dict]:
  for i in infos:
    if i["healthy"]:
      return i
  return None


# --------------------------------------------------------------------------
# creation + verification
# --------------------------------------------------------------------------
def create_and_verify(cfg: dict, name: str) -> tuple:
  """Returns (ok, detail). Never deletes anything except its own probe file."""
  ws = os.path.join(cfg["citc_root"], name)
  g3 = os.path.join(ws, "google3")
  cmd = shlex.split(cfg["g4"]) + ["citc", "-c", cfg["cl"], name]
  t0 = time.time()
  try:
    p = subprocess.run(cmd, cwd=os.path.expanduser("~"), capture_output=True,
                       text=True, timeout=180)
  except (OSError, subprocess.TimeoutExpired) as e:
    return False, f"`{' '.join(cmd)}` failed: {e!r}"
  out = (p.stdout + p.stderr).strip().replace("\n", " | ")
  if p.returncode != 0:
    return False, f"`{' '.join(cmd)}` rc={p.returncode}: {out[-400:]}"
  log(cfg, f"created {name} in {time.time() - t0:.1f}s: {out[-200:]}")

  cl = baseline_cl(ws)
  if cl != cfg["cl"]:
    return False, f"baseline CL read back {cl!r} != {cfg['cl']}"
  tv = torch_version(g3)
  if not torch_ge(tv, cfg["min_torch"]):
    return False, f"torch METADATA {tv!r} < {cfg['min_torch']}"
  stages = os.path.join(g3, "experimental/qiaos/eqr_jax_final_stages")
  try:
    os.makedirs(stages, exist_ok=True)
  except OSError as e:
    return False, f"mkdir {stages}: {e!r}"
  probe = os.path.join(stages, f".stage_ws_probe_{os.getpid()}")
  token = f"probe {name} pid={os.getpid()} t={time.time():.3f}\n"
  try:
    with open(probe, "w", encoding="utf-8") as f:
      f.write(token)
      f.flush()
      os.fsync(f.fileno())
    for _ in range(2):  # read back after 6 s and again after 12 s
      time.sleep(cfg["probe_sleep"])
      with open(probe, encoding="utf-8") as f:
        got = f.read()
      if got != token:
        return False, f"probe read back {got!r} != {token!r}"
  except OSError as e:
    return False, f"probe write/read: {e!r}"
  finally:
    try:
      os.remove(probe)
    except OSError:
      pass
  dr = _size(os.path.join(ws, ".citc/dropped_resources.ascii"))
  if not 0 <= dr <= _DROPPED_HEADER_BYTES:
    return False, f"dropped_resources.ascii={dr}B after probe (writes dropped)"
  mf = _size(os.path.join(ws, ".citc/manifest"))
  return True, f"cl={cl} torch={tv} manifest={mf} dropped={dr}"


# --------------------------------------------------------------------------
# ensure
# --------------------------------------------------------------------------
class _Lock:
  def __init__(self, path: str):
    self.path = path
    self.fd = None

  def acquire(self, wait_s: float) -> bool:
    self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + max(0.0, wait_s)
    while True:
      try:
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, f"{os.getpid()} {_now()}\n".encode())
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


def _rails(cfg: dict, family: str) -> Optional[str]:
  """Refusal reason from the cooldown / daily cap, or None."""
  st = _load_state(cfg).get(family, {})
  now = time.time()
  last_fail = float(st.get("last_failure_ts", 0) or 0)
  if now - last_fail < cfg["fail_cooldown"]:
    return (f"cooldown: last creation failure {int(now - last_fail)}s ago "
            f"(< {cfg['fail_cooldown']}s): {st.get('last_failure', '')}")
  recent = [t for t in st.get("created_ts", []) if now - float(t) < 86400]
  if len(recent) >= cfg["max_per_day"]:
    return (f"daily cap: {len(recent)} creation(s) in the last 24h "
            f"(TPU_STAGE_WS_MAX_CREATES_PER_DAY={cfg['max_per_day']})")
  return None


def _record(cfg: dict, family: str, *, created: Optional[str] = None,
            failure: Optional[str] = None) -> None:
  state = _load_state(cfg)
  st = state.setdefault(family, {})
  now = time.time()
  if created:
    st["created_ts"] = [t for t in st.get("created_ts", [])
                        if now - float(t) < 86400] + [now]
    st.setdefault("created", []).append({"name": created, "at": _now()})
  if failure:
    st["last_failure_ts"] = now
    st["last_failure"] = failure[:500]
  _save_state(cfg, state)


def _summary(infos: list) -> str:
  return ", ".join(
      f"{i['name']}(mf={i['manifest']},torch={i['torch'] or '?'},"
      f"{'ok' if i['healthy'] else '/'.join(i['reasons'])})" for i in infos
  ) or "none"


def _daemonize(cfg: dict) -> bool:
  """Double-fork; returns True in the detached child, False in the caller."""
  pid = os.fork()
  if pid > 0:
    os.waitpid(pid, 0)
    return False
  os.setsid()
  if os.fork() > 0:
    os._exit(0)
  # Leave the caller's cwd (often a CitC FUSE path whose handle dies when the
  # mount is recreated).
  os.chdir(os.path.expanduser("~"))
  # Detach from the caller's pipes (the dispatch worker reads `tpu queue`'s
  # stdout to EOF; an inherited fd would hold that read open).
  devnull = os.open(os.devnull, os.O_RDWR)
  try:
    logfd = os.open(cfg["log"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
  except OSError:
    logfd = devnull
  os.dup2(devnull, 0)
  os.dup2(devnull, 1)
  os.dup2(logfd, 2)  # tracebacks land in the log
  os.closerange(3, 1024)
  global _DETACHED
  _DETACHED = True
  return True


def _heartbeat(cfg: dict, family: str, pref: dict) -> None:
  """Proof the hook is INVOKED: no-op checks are not logged (one per stage),
  so overwrite one line per family with the last check instead."""
  path = f"{cfg['heartbeat']}.{family}"
  try:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
      f.write(f"{_now()} pid={os.getpid()} {family} ok {pref['name']} "
              f"manifest={pref['manifest']}\n")
    os.replace(tmp, path)
  except OSError:
    pass


def ensure(cfg: dict, family: str, *, dry_run: bool = False,
           background: bool = False, wait_s: float = 0.0) -> int:
  infos = scan(cfg, family)
  if any(i["headroom"] for i in infos):
    pref = _preferred(infos)
    print(f"[stage-ws-rotate] {family}: ok, {pref['name']} has headroom "
          f"(manifest {pref['manifest']} < {cfg['precreate']})", file=sys.stderr)
    if not dry_run:
      _heartbeat(cfg, family, pref)
    print(pref["google3"])
    return 0
  name = next_name(cfg, family)
  why = (f"{family}: no member below {cfg['precreate']}B with torch >= "
         f"{cfg['min_torch']} [{_summary(infos)}]")
  if dry_run:
    log(cfg, f"DRY RUN {why}; would create {name} at CL {cfg['cl']}",
        to_file=False)
    pref = _preferred(infos)
    if pref:
      print(pref["google3"])
    return 0
  refusal = _rails(cfg, family)
  if refusal:
    log(cfg, f"{why}; NOT creating ({refusal})")
    return 1
  if background:
    log(cfg, f"{why}; creating {name} in the background")
    if not _daemonize(cfg):
      return 0
    wait_s = 0.0  # a background creator never queues behind another creator
  lock = _Lock(cfg["lock"])
  try:
    if not lock.acquire(wait_s):
      log(cfg, f"{family}: lock {cfg['lock']} busy (another creator); "
          f"not creating")
      return 2
    infos = scan(cfg, family)  # re-check under the lock: idempotent
    if any(i["headroom"] for i in infos):
      pref = _preferred(infos)
      log(cfg, f"{family}: {pref['name']} appeared while waiting for the "
          f"lock; nothing to do")
      print(pref["google3"])
      return 0
    refusal = _rails(cfg, family)
    if refusal:
      log(cfg, f"{family}: NOT creating ({refusal})")
      return 1
    name = next_name(cfg, family)
    log(cfg, f"{family}: creating {name} at CL {cfg['cl']} "
        f"[{_summary(infos)}]")
    ok, detail = create_and_verify(cfg, name)
    if not ok:
      log(cfg, f"{family}: {name} FAILED verification: {detail}; "
          f"listed in {cfg['bad']} (kept, not deleted)")
      if os.path.isdir(os.path.join(cfg["citc_root"], name)):
        with open(cfg["bad"], "a", encoding="utf-8") as f:
          f.write(name + "\n")
      _record(cfg, family, failure=f"{name}: {detail}")
      return 1
    _record(cfg, family, created=name)
    log(cfg, f"{family}: {name} VERIFIED ({detail}); joins the "
        f"{family} pool")
    print(os.path.join(cfg["citc_root"], name, "google3"))
    return 0
  finally:
    lock.release()
    if background:
      sys.stdout.flush()
      os._exit(0)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  sub = ap.add_subparsers(dest="cmd", required=True)
  s = sub.add_parser("status")
  s.add_argument("--family", default="all",
                 choices=sorted(FAMILIES) + ["all"])
  s.add_argument("--json", action="store_true")
  e = sub.add_parser("ensure")
  e.add_argument("--family", required=True, choices=sorted(FAMILIES))
  e.add_argument("--dry-run", "--dry_run", action="store_true")
  e.add_argument("--background", action="store_true")
  e.add_argument("--wait", type=float, default=0.0)
  a = ap.parse_args(argv)
  cfg = _cfg()
  if a.cmd == "status":
    fams = sorted(FAMILIES) if a.family == "all" else [a.family]
    res = {f: scan(cfg, f) for f in fams}
    if a.json:
      print(json.dumps(res, indent=1))
    else:
      for f in fams:
        print(f"{f} ({FAMILIES[f]}*, CL {cfg['cl']}, limit {cfg['max_manifest']},"
              f" precreate at {cfg['precreate']}):")
        for i in res[f] or [{"name": "(none)"}]:
          if "google3" not in i:
            print("  (none)")
            continue
          print(f"  {i['name']:<34} manifest={i['manifest']:<9} "
                f"dropped={i['dropped']:<4} cl={i['cl']:<10} "
                f"torch={i['torch'] or '?':<9} "
                f"{'HEADROOM' if i['headroom'] else 'healthy' if i['healthy'] else 'UNHEALTHY ' + '/'.join(i['reasons'])}")
    return 0
  return ensure(cfg, a.family, dry_run=a.dry_run, background=a.background,
                wait_s=a.wait)


if __name__ == "__main__":
  sys.exit(main())
