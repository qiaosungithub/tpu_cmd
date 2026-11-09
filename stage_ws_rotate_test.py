#!/usr/bin/env python3
"""Self-asserting tests for stage_ws_rotate.py. Run: python3 stage_ws_rotate_test.py

Everything runs against a FAKE CitC root in a temp dir and a fake `g4` script,
so no real workspace is ever created. Each test first checks the negative
control (the verdict flips when the property is broken).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import contextlib

import stage_ws_rotate as r

_fail = []

FAKE_G4 = r'''#!/bin/bash
# fake `g4 citc -c <CL> <name>`: builds a minimal CitC-looking tree.
# FAKE_G4_MODE: ok | fail | wrongcl | oldtorch | dropwrites
[ "$1" = citc ] && [ "$2" = -c ] || { echo "bad args: $*" >&2; exit 9; }
cl="$3"; name="$4"; root="$FAKE_CITC_ROOT/$name"
echo "$(date +%s.%N) $name" >> "$FAKE_CITC_ROOT/.g4_calls"
case "${FAKE_G4_MODE:-ok}" in fail) echo "simulated failure" >&2; exit 1;; esac
[ "${FAKE_G4_MODE}" = wrongcl ] && cl=111
mkdir -p "$root/.citc" "$root/google3/third_party/py/torch"
printf 'workspace {\n  change: %s\n}\n' "$cl" > "$root/.citc/srcfs_workspace.ascii"
head -c 79 /dev/zero > "$root/.citc/manifest"
printf '%-48s\n' '# devtools_srcfs.Workspace DroppedLocalResources' | head -c 49 > "$root/.citc/dropped_resources.ascii"
v=2.15.0a0; [ "${FAKE_G4_MODE}" = oldtorch ] && v=2.13.0
printf 'third_party {\n  version: "9b6e45278f06"\n  version: "%s"\n}\n' "$v" > "$root/google3/third_party/py/torch/METADATA"
if [ "${FAKE_G4_MODE}" = dropwrites ]; then echo x >> "$root/.citc/dropped_resources.ascii"; fi
echo "Created CITC workspace (synced @$cl)"
'''


def _ck(cond, msg):
  if cond:
    print(f"  ok   {msg}")
  else:
    print(f"  FAIL {msg}")
    _fail.append(msg)


class Env:
  """Temp CitC root + state dir + fake g4, wired through the env knobs."""

  def __init__(self, **extra):
    self.tmp = tempfile.mkdtemp(prefix="stage_ws_rotate_test_")
    self.root = os.path.join(self.tmp, "cloud")
    self.state = os.path.join(self.tmp, "state")
    os.makedirs(self.root)
    os.makedirs(self.state)
    self.g4 = os.path.join(self.tmp, "g4")
    with open(self.g4, "w") as f:
      f.write(FAKE_G4)
    os.chmod(self.g4, 0o755)
    self.env = {
        "TPU_STAGE_WS_CITC_ROOT": self.root,
        "TPU_STAGE_WS_ROTATE_DIR": self.state,
        "TPU_STAGE_WS_G4": self.g4,
        "TPU_STAGE_WS_PROBE_SLEEP_S": "0.05",
        "FAKE_CITC_ROOT": self.root,
        **extra,
    }
    self._saved = {}

  def __enter__(self):
    for k, v in self.env.items():
      self._saved[k] = os.environ.get(k)
      os.environ[k] = v
    return self

  def __exit__(self, *a):
    for k, v in self._saved.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v
    shutil.rmtree(self.tmp, ignore_errors=True)

  def make(self, name, manifest=79, torch="2.15.0a0", cl="989229158",
           dropped=49):
    ws = os.path.join(self.root, name)
    os.makedirs(os.path.join(ws, ".citc"), exist_ok=True)
    os.makedirs(os.path.join(ws, "google3/third_party/py/torch"), exist_ok=True)
    with open(os.path.join(ws, ".citc/manifest"), "wb") as f:
      f.truncate(manifest)
    with open(os.path.join(ws, ".citc/dropped_resources.ascii"), "wb") as f:
      f.write(b"#" * dropped)
    with open(os.path.join(ws, ".citc/srcfs_workspace.ascii"), "w") as f:
      f.write(f"change: {cl}\n")
    with open(os.path.join(ws, "google3/third_party/py/torch/METADATA"), "w") as f:
      f.write(f'  version: "abcdef0123"\n  version: "{torch}"\n')

  def calls(self):
    try:
      with open(os.path.join(self.root, ".g4_calls")) as f:
        return [ln.split()[1] for ln in f if ln.strip()]
    except OSError:
      return []

  def logtext(self):
    try:
      with open(os.path.join(self.state, ".tpu_stage_ws_rotate.log")) as f:
        return f.read()
    except OSError:
      return ""


def _run(*argv):
  out = io.StringIO()
  with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    rc = r.main(list(argv))
  return rc, out.getvalue().strip()


def test_version_parsing():
  print("version parsing")
  _ck(r.torch_ge("2.15.0a0", "2.15"), "2.15.0a0 >= 2.15")
  _ck(r.torch_ge("2.16.0a0", "2.15"), "2.16 >= 2.15")
  _ck(r.torch_ge("3.0", "2.15"), "3.0 >= 2.15")
  _ck(not r.torch_ge("2.13.0", "2.15"), "2.13 < 2.15 (negative control)")
  _ck(not r.torch_ge("", "2.15"), "unknown version is not >= 2.15")
  with Env() as e:
    e.make("x", torch="2.15.0a0")
    _ck(r.torch_version(os.path.join(e.root, "x/google3")) == "2.15.0a0",
        "commit-hash `version:` line is skipped, semantic one read")


def test_naming_and_order():
  print("naming and order")
  with Env() as e:
    cfg = r._cfg()
    _ck(r.next_name(cfg, "qiaos", "20261109") == "tpu_staging_t215_20261109",
        "first name is <prefix><date>")
    e.make("tpu_staging_t215_20261109")
    _ck(r.next_name(cfg, "qiaos", "20261109") == "tpu_staging_t215_20261109_2",
        "same-day collision gets _2")
    e.make("tpu_staging_t215_20261109_2")
    e.make("tpu_staging_t215_20261109_10")
    e.make("tpu_staging_t215_20261108")
    e.make("tpu_staging_lyy_t215_20261109")
    e.make("tpu_staging_t215_junk")
    m = r.members(cfg, "qiaos")
    _ck(m == ["tpu_staging_t215_20261108", "tpu_staging_t215_20261109",
              "tpu_staging_t215_20261109_2", "tpu_staging_t215_20261109_10"],
        f"qiaos members oldest first, numeric suffix order, no lyy/junk: {m}")
    _ck(r.members(cfg, "lyy") == ["tpu_staging_lyy_t215_20261109"],
        "lyy family is disjoint from qiaos")


def test_health():
  print("health")
  with Env() as e:
    cfg = r._cfg()
    e.make("a", manifest=100)
    _ck(r.inspect(cfg, "a")["headroom"], "small fresh ws has headroom")
    e.make("b", manifest=9437184)
    i = r.inspect(cfg, "b")
    _ck(i["healthy"] and not i["headroom"], "9 MiB: healthy, no headroom")
    e.make("c", manifest=10485760)
    _ck(not r.inspect(cfg, "c")["healthy"], "10 MiB: unhealthy (wrapper limit)")
    e.make("d", torch="2.13.0")
    _ck(not r.inspect(cfg, "d")["healthy"], "torch 2.13: unhealthy")
    e.make("f", dropped=60)
    _ck(not r.inspect(cfg, "f")["healthy"], "dropped writes: unhealthy")
    with open(os.path.join(e.state, ".tpu_stage_ws_bad"), "w") as f:
      f.write("a\n")
    _ck(not r.inspect(cfg, "a")["healthy"], "listed-bad: unhealthy")


def test_ensure_noop_when_headroom():
  print("ensure is a no-op when a member has headroom")
  with Env() as e:
    e.make("tpu_staging_t215_20261109", manifest=1000)
    rc, out = _run("ensure", "--family", "qiaos")
    _ck(rc == 0 and out.endswith("tpu_staging_t215_20261109/google3"),
        f"rc=0 and prints the member ({rc}, {out!r})")
    _ck(e.calls() == [], "g4 never called")
    hb = os.path.join(e.state, ".tpu_stage_ws_rotate.heartbeat.qiaos")
    _ck(os.path.exists(hb) and "ok tpu_staging_t215_20261109" in open(hb).read(),
        "no-op check leaves a heartbeat (proof of invocation)")
    os.remove(hb)
    _run("ensure", "--family", "qiaos", "--dry-run")
    _ck(not os.path.exists(hb), "dry run writes no heartbeat")


def test_ensure_creates_when_full():
  print("ensure creates + verifies when every member is full")
  with Env() as e:
    e.make("tpu_staging_t215_20261101", manifest=10485760)  # over limit
    e.make("tpu_staging_t215_20261105", manifest=9500000)   # healthy, no headroom
    rc, out = _run("ensure", "--family", "qiaos", "--dry-run")
    _ck(rc == 0 and e.calls() == [], "dry run creates nothing")
    _ck(out.endswith("tpu_staging_t215_20261105/google3"),
        "dry run still prints the healthy member")
    rc, out = _run("ensure", "--family", "qiaos")
    today = time.strftime("%Y%m%d", time.gmtime())
    want = f"tpu_staging_t215_{today}"
    _ck(rc == 0 and e.calls() == [want], f"created {want} ({rc}, {e.calls()})")
    _ck(out.endswith(f"{want}/google3"), "prints the new member")
    _ck(os.path.isdir(os.path.join(
        e.root, want, "google3/experimental/qiaos/eqr_jax_final_stages")),
        "eqr_jax_final_stages created")
    _ck(not [n for n in os.listdir(os.path.join(
        e.root, want, "google3/experimental/qiaos/eqr_jax_final_stages"))],
        "probe file removed")
    _ck("VERIFIED" in e.logtext(), "creation logged as VERIFIED")
    # idempotent: a second call finds the new member and does nothing
    rc, _ = _run("ensure", "--family", "qiaos")
    _ck(rc == 0 and len(e.calls()) == 1, "second ensure is a no-op")
    # old members untouched
    _ck(os.path.getsize(os.path.join(
        e.root, "tpu_staging_t215_20261101/.citc/manifest")) == 10485760,
        "full member left in place, untouched")


def test_ensure_creates_when_none():
  print("ensure creates when the family is empty (lyy bootstrap)")
  with Env() as e:
    rc, out = _run("ensure", "--family", "lyy")
    _ck(rc == 0 and len(e.calls()) == 1 and
        e.calls()[0].startswith("tpu_staging_lyy_t215_"),
        f"lyy member created ({e.calls()})")


def test_verification_failures():
  print("verification failures are listed bad, recorded, and cool down")
  for mode, needle in (("wrongcl", "baseline CL"), ("oldtorch", "torch METADATA"),
                       ("dropwrites", "dropped_resources"), ("fail", "rc=1")):
    with Env(FAKE_G4_MODE=mode) as e:
      rc, _ = _run("ensure", "--family", "qiaos")
      _ck(rc == 1, f"{mode}: rc=1")
      _ck(needle in e.logtext(), f"{mode}: log names the cause ({needle})")
      bad = os.path.join(e.state, ".tpu_stage_ws_bad")
      if mode == "fail":
        _ck(not os.path.exists(bad), "fail: nothing to list (no ws created)")
      else:
        _ck(os.path.exists(bad) and open(bad).read().strip(),
            f"{mode}: created ws listed bad")
        names = r.members(r._cfg(), "qiaos")
        _ck(names and not r.inspect(r._cfg(), names[0])["healthy"],
            f"{mode}: listed ws is not healthy")
      n = len(e.calls())
      rc, _ = _run("ensure", "--family", "qiaos")
      _ck(rc == 1 and len(e.calls()) == n and "cooldown" in e.logtext(),
          f"{mode}: retry inside cooldown does not call g4 again")


def test_daily_cap():
  print("daily cap")
  with Env(TPU_STAGE_WS_MAX_CREATES_PER_DAY="2",
           TPU_STAGE_WS_PRECREATE_BYTES="1") as e:
    # precreate=1 byte: nothing ever has headroom -> would create every call
    for _ in range(4):
      _run("ensure", "--family", "qiaos")
    _ck(len(e.calls()) == 2, f"capped at 2 creations ({e.calls()})")
    _ck("daily cap" in e.logtext(), "cap refusal logged")


def test_lock_busy_and_concurrency():
  print("lock: busy -> rc 2; concurrent callers create exactly one")
  with Env() as e:
    cfg = r._cfg()
    holder = r._Lock(cfg["lock"])
    _ck(holder.acquire(0), "test holds the lock")
    rc, _ = _run("ensure", "--family", "qiaos")
    _ck(rc == 2 and e.calls() == [], "busy lock, wait 0 -> rc 2, no g4")
    holder.release()
    # 4 real processes race; the slow fake g4 widens the window
    env = dict(os.environ)
    procs = [subprocess.Popen(
        [sys.executable, r.__file__, "ensure", "--family", "qiaos",
         "--wait", "30"], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE) for _ in range(4)]
    rcs = [p.wait(60) for p in procs]
    _ck(rcs == [0, 0, 0, 0], f"all racers rc 0 ({rcs})")
    _ck(len(e.calls()) == 1, f"exactly one creation ({e.calls()})")


def test_background():
  print("background: returns at once, child creates, caller's pipe closes")
  with Env(TPU_STAGE_WS_PROBE_SLEEP_S="1") as e:
    t0 = time.time()
    p = subprocess.run([sys.executable, r.__file__, "ensure", "--family",
                        "qiaos", "--background"], env=dict(os.environ),
                       capture_output=True, text=True, timeout=30)
    dt = time.time() - t0
    # capture_output reads to EOF: if the child kept the pipe, this would
    # take >= 2 s (two 1 s probe sleeps)
    _ck(p.returncode == 0 and dt < 1.5,
        f"caller returns in {dt:.2f}s with rc {p.returncode}")
    for _ in range(100):
      if "VERIFIED" in e.logtext():
        break
      time.sleep(0.1)
    _ck("VERIFIED" in e.logtext() and len(e.calls()) == 1,
        "detached child created and verified the member")
    _ck(e.logtext().count("VERIFIED") == 1, "no duplicate log lines")


def main():
  for t in (test_version_parsing, test_naming_and_order, test_health,
            test_ensure_noop_when_headroom, test_ensure_creates_when_full,
            test_ensure_creates_when_none, test_verification_failures,
            test_daily_cap, test_lock_busy_and_concurrency, test_background):
    t()
  print()
  if _fail:
    print(f"{len(_fail)} FAILURE(S):")
    for f in _fail:
      print("  -", f)
    sys.exit(1)
  print("ALL PASSED")


if __name__ == "__main__":
  main()
