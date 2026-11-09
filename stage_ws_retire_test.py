#!/usr/bin/env python3
"""Self-asserting tests for stage_ws_retire.py. Run: python3 stage_ws_retire_test.py

Everything runs against a FAKE CitC root, fake record files, a fake `g4`, a fake
`fileutil` (a local directory stands in for /cns) and a fake `pgrep`, all in a
temp dir, so no real workspace, record or CNS path is read or written. Each
rule is checked together with its negative control (the verdict flips when only
that property changes).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import stage_ws_retire as ret
import stage_ws_rotate as rot

_fail = []
DAY = 86400
MIB = 1048576

FAKE_G4 = r'''#!/bin/bash
# fake g4: `citc -c <CL> <name>` builds a minimal CitC tree; `citc -d <name>`
# removes it. Every call is appended to $FAKE_EVENTS.
echo "g4 $*" >> "$FAKE_EVENTS"
[ "$1" = citc ] || { echo "bad args: $*" >&2; exit 9; }
if [ "$2" = -d ]; then
  [ "${FAKE_G4_DELETE_MODE:-ok}" = fail ] && { echo "simulated delete failure" >&2; exit 1; }
  rm -rf "$FAKE_CITC_ROOT/$3"; echo "deleted $3"; exit 0
fi
[ "$2" = -c ] || { echo "bad args: $*" >&2; exit 9; }
root="$FAKE_CITC_ROOT/$4"
mkdir -p "$root/.citc" "$root/google3/third_party/py/torch"
printf 'workspace {\n  change: %s\n}\n' "$3" > "$root/.citc/srcfs_workspace.ascii"
head -c 79 /dev/zero > "$root/.citc/manifest"
head -c 49 /dev/zero > "$root/.citc/dropped_resources.ascii"
printf 'third_party {\n  version: "2.15.0a0"\n}\n' > "$root/google3/third_party/py/torch/METADATA"
echo "Created CITC workspace $4"
'''

FAKE_FILEUTIL = r'''#!/bin/bash
# fake fileutil: /cns/... maps to $FAKE_CNS/... . Supports `mkdir -p D` and
# `cp -f SRC DST`. FAKE_FU_MODE injects failures.
echo "fileutil $*" >> "$FAKE_EVENTS"
map() { case "$1" in /cns/*) echo "$FAKE_CNS/${1#/cns/}";; *) echo "$1";; esac; }
mode="${FAKE_FU_MODE:-ok}"
case "$1" in
  mkdir)
    [ "$mode" = fail_mkdir ] && { echo "simulated mkdir failure" >&2; exit 1; }
    mkdir -p "$(map "$3")"; exit $?;;
  cp)
    src="$3"; dst="$4"
    case "$dst" in
      /cns/*.tar.gz) [ "$mode" = fail_cp_tar ] && { echo "simulated cp failure" >&2; exit 1; }
                     if [ "$mode" = corrupt ]; then head -c 10 "$src" > "$(map "$dst")"; exit 0; fi;;
      /cns/*INDEX.jsonl) [ "$mode" = fail_index ] && { echo "simulated index failure" >&2; exit 1; };;
    esac
    case "$src" in /cns/*) [ "$mode" = fail_readback ] && { echo "simulated read failure" >&2; exit 1; };; esac
    cp -f "$(map "$src")" "$(map "$dst")"; exit $?;;
esac
echo "fake fileutil: unsupported $*" >&2; exit 9
'''

FAKE_PGREP = r'''#!/bin/bash
# fake pgrep -c fileutil: pops the first line of $FAKE_PGREP_SEQ (default 0).
echo "pgrep $*" >> "$FAKE_EVENTS"
[ "${FAKE_PGREP_MODE:-ok}" = error ] && exit 3
f="$FAKE_PGREP_SEQ"
if [ -n "$f" ] && [ -s "$f" ]; then
  n=$(head -n1 "$f"); [ "$(wc -l < "$f")" -gt 1 ] && sed -i 1d "$f"
  echo "$n"; [ "$n" = 0 ] && exit 1; exit 0
fi
echo 0; exit 1
'''


def _ck(cond, msg):
  if cond:
    print(f"  ok   {msg}")
  else:
    print(f"  FAIL {msg}")
    _fail.append(msg)


def _script(path, body):
  with open(path, "w") as f:
    f.write(body)
  os.chmod(path, 0o755)
  return path


class Env:
  """Temp CitC root, records, fake CNS and fake tools, wired through env."""

  def __init__(self, **extra):
    self.tmp = tempfile.mkdtemp(prefix="stage_ws_retire_test_")
    j = lambda *p: os.path.join(self.tmp, *p)  # noqa: E731
    self.root, self.state, self.home, self.cns = (j("cloud"), j("state"),
                                                  j("home"), j("cns"))
    for d in (self.root, self.state, self.home, self.cns, j("scratch")):
      os.makedirs(d)
    self.events = j("events")
    open(self.events, "w").close()
    self.queue, self.jobs = j("home", "queue.json"), j("home", "jobs.json")
    self.legacy, self.prov = j("home", "legacy.json"), j("home", "prov.jsonl")
    self.pgrep_seq = j("pgrep_seq")
    self.env = {
        "TPU_STAGE_WS_CITC_ROOT": self.root,
        "TPU_STAGE_WS_ROTATE_DIR": self.state,
        "TPU_STAGE_WS_G4": _script(j("g4"), FAKE_G4),
        "TPU_STAGE_WS_FILEUTIL": _script(j("fileutil"), FAKE_FILEUTIL),
        "TPU_STAGE_WS_PGREP": _script(j("pgrep"), FAKE_PGREP),
        "TPU_STAGE_WS_FILEUTIL_WAIT_S": "0.01",
        "TPU_STAGE_WS_FILEUTIL_WAIT_MAX_S": "0.05",
        "TPU_STAGE_WS_STAGE_LOCK": j("stage.lock"),
        "TPU_STAGE_WS_STAGE_LOCK_WAIT_S": "2",
        "TPU_STAGE_WS_RETIRE_TMP": j("scratch"),
        "TPU_STAGE_WS_ARCHIVE_ROOT": "/cns/zz-d/home/qiaos/eqr_data/stage_ws_retired",
        "TPU_STAGE_WS_RETIRE_QUEUES": self.queue,
        "TPU_STAGE_WS_RETIRE_REGISTRIES": self.jobs,
        "TPU_STAGE_WS_RETIRE_AGED": os.pathsep.join([self.legacy, self.prov]),
        "TPU_STAGE_WS_RETIRE_CHECK": "1",
        "FAKE_CITC_ROOT": self.root,
        "FAKE_CNS": self.cns,
        "FAKE_EVENTS": self.events,
        "FAKE_PGREP_SEQ": self.pgrep_seq,
        **extra,
    }
    self._saved = {}

  def __enter__(self):
    for k, v in self.env.items():
      self._saved[k] = os.environ.get(k)
      os.environ[k] = v
    self.write_records()
    return self

  def __exit__(self, *a):
    for k, v in self._saved.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v
    shutil.rmtree(self.tmp, ignore_errors=True)

  # -- fixtures ------------------------------------------------------------
  def member(self, name, manifest, stagedirs, parent_age_d=None, torch="2.15.0a0"):
    """stagedirs: {basename: age_days}. Creates a CitC-looking client."""
    ws = os.path.join(self.root, name)
    g3 = os.path.join(ws, "google3")
    os.makedirs(os.path.join(ws, ".citc"))
    os.makedirs(os.path.join(g3, "third_party/py/torch"))
    with open(os.path.join(ws, ".citc/srcfs_workspace.ascii"), "w") as f:
      f.write("workspace {\n  change: 989229158\n}\n")
    with open(os.path.join(ws, ".citc/manifest"), "wb") as f:
      f.truncate(manifest)
    with open(os.path.join(ws, ".citc/dropped_resources.ascii"), "wb") as f:
      f.write(b"#" * 49)
    with open(os.path.join(g3, "third_party/py/torch/METADATA"), "w") as f:
      f.write(f'third_party {{\n  version: "{torch}"\n}}\n')
    parent = os.path.join(g3, ret.STAGES_REL)
    os.makedirs(parent)
    now = time.time()
    for b, age in stagedirs.items():
      d = os.path.join(parent, b)
      os.makedirs(os.path.join(d, "pkg"))
      with open(os.path.join(d, "pkg", "main.py"), "w") as f:
        f.write(f"# stagedir {b} of {name}\nprint('hi')\n")
      os.utime(d, (now - age * DAY, now - age * DAY))
    pa = parent_age_d if parent_age_d is not None else (
        max(stagedirs.values()) if stagedirs else 60)
    os.utime(parent, (now - pa * DAY, now - pa * DAY))
    return ws

  def manual(self, name, manifest=11 * MIB):
    return self.member(name, manifest, {"eqr_run_ca_manual01": 90})

  def write_records(self, queue=(), jobs=None, legacy=None, prov=(),
                    legacy_mtime_age_d=None):
    with open(self.queue, "w") as f:
      json.dump({"schema_version": 1, "entries": list(queue)}, f)
    with open(self.jobs, "w") as f:
      json.dump(jobs or {}, f)
    with open(self.legacy, "w") as f:
      json.dump(legacy or {}, f)
    if legacy_mtime_age_d is not None:
      t = time.time() - legacy_mtime_age_d * DAY
      os.utime(self.legacy, (t, t))
    with open(self.prov, "w") as f:
      for p in prov:
        f.write(json.dumps(p) + "\n")

  def events_list(self):
    with open(self.events) as f:
      return [ln.rstrip("\n") for ln in f]


def iso_ago(days):
  return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - days * DAY))


def rel(b):
  return f"{ret.STAGES_REL}/{b}"


A = "tpu_staging_t215_20260901"
B = "tpu_staging_t215_20261001"
N = "tpu_staging_t215_20261109"
LA = "tpu_staging_lyy_t215_20260901"
LN = "tpu_staging_lyy_t215_20261109"
XID_OLD = "300000001"


def base_family(e, a_manifest=int(10.2 * MIB), a_stage_age=60, n_manifest=MIB):
  e.member(A, a_manifest, {"eqr_run_ca_aaa111": a_stage_age,
                           "eqr_run_ca_unref1": a_stage_age})
  e.member(N, n_manifest, {"eqr_run_ca_nnn111": 0.1})


def old_legacy(days=60, b="eqr_run_ca_aaa111", xid=XID_OLD):
  return {xid: {"status": "COMPLETED", "stagedir": rel(b),
                "archived_at": iso_ago(days)}}


def verdict(name):
  p = ret.plan(ret.cfg())
  m = next((x for x in p["members"] if x["name"] == name), None)
  return p, m


# --------------------------------------------------------------------------
def test_rules():
  print("test_rules")
  # Baseline: A is retirable, N (newest healthy) is not.
  with Env() as e:
    base_family(e)
    e.write_records(legacy=old_legacy())
    p, m = verdict(A)
    _ck(m["retirable"], f"baseline: full idle old-referenced {A} is RETIRABLE")
    _ck(sorted(m["archive"]) == ["eqr_run_ca_aaa111"] and m["unreferenced"] == 1,
        "baseline: archive set = the old-referenced stagedir only; 1 unreferenced")
    _, n = verdict(N)
    _ck(not n["retirable"] and "newest healthy member" in n["reasons"],
        "the newest healthy member is protected")
    _ck(p["complete"] and not p["errors"], "baseline plan is complete")

  # Rule 1: newest healthy member is protected even when full and idle.
  with Env() as e:
    base_family(e, n_manifest=int(9.6 * MIB))
    e.member(B, int(9.7 * MIB), {"eqr_run_ca_bbb111": 50})
    os.utime(os.path.join(e.root, N, "google3", ret.STAGES_REL, "eqr_run_ca_nnn111"),
             (time.time() - 50 * DAY,) * 2)
    os.utime(os.path.join(e.root, N, "google3", ret.STAGES_REL),
             (time.time() - 50 * DAY,) * 2)
    e.write_records(legacy=old_legacy())
    _, n = verdict(N)
    _, b = verdict(B)
    _ck(not n["retirable"] and n["reasons"] == ["newest healthy member"],
        "rule 1: a 9.6 MiB idle newest healthy member is kept for that reason alone")
    _ck(b["retirable"], "rule 1 control: an older healthy 9.7 MiB idle member IS retirable")
  with Env() as e:
    e.member(A, int(10.2 * MIB), {"eqr_run_ca_aaa111": 60})  # only member, unhealthy
    e.write_records(legacy=old_legacy())
    _, m = verdict(A)
    _ck(not m["retirable"] and any("no member is healthy" in r for r in m["reasons"]),
        "rule 1: with no healthy member the newest member is protected")

  # Rule 2: manifest >= 9 MiB.
  with Env() as e:
    base_family(e, a_manifest=8 * MIB)
    e.write_records(legacy=old_legacy())
    _, m = verdict(A)
    _ck(not m["retirable"] and any("not rotated out" in r for r in m["reasons"]),
        "rule 2: an 8 MiB member is not retirable")
  with Env() as e:
    base_family(e, a_manifest=9 * MIB)
    e.write_records(legacy=old_legacy())
    _, m = verdict(A)
    _ck(m["retirable"], "rule 2 control: exactly 9 MiB is retirable")

  # Rule 3: last stage > 30 days (stagedir mtime, parent mtime, provenance).
  with Env() as e:
    base_family(e, a_stage_age=10)
    e.write_records(legacy=old_legacy())
    _, m = verdict(A)
    _ck(not m["retirable"] and any("last stage" in r for r in m["reasons"]),
        "rule 3: stagedirs staged 10 days ago block")
  with Env() as e:
    base_family(e)
    os.utime(os.path.join(e.root, A, "google3", ret.STAGES_REL),
             (time.time() - 3 * DAY,) * 2)
    e.write_records(legacy=old_legacy())
    _, m = verdict(A)
    _ck(not m["retirable"], "rule 3: a staging parent touched 3 days ago blocks")
  for age, want in ((5, False), (40, True)):
    with Env() as e:
      base_family(e)
      absdir = os.path.join(e.root, A, "google3", rel("eqr_run_ca_aaa111"))
      e.write_records(prov=[{"ts": iso_ago(age) + "Z", "dir": absdir,
                             "rel": rel("eqr_run_ca_aaa111"), "reused": True}])
      _, m = verdict(A)
      _ck(m["retirable"] == want,
          f"rule 3: a CA re-use (provenance only) {age}d ago -> retirable={want}")
      if want:
        _ck(sorted(m["archive"]) == ["eqr_run_ca_aaa111"],
            "rule 4: an old provenance record puts the stagedir in the archive set")

  # Rule 4: references.
  cases = [
      ("queue entry via prior_xids (resolved through the legacy registry)",
       dict(queue=[{"job_id": "q1", "state": "DONE", "xid": "399",
                    "prior_xids": [XID_OLD]}], legacy=old_legacy()), False),
      ("control: a queue entry naming an unrelated XID",
       dict(queue=[{"job_id": "q1", "state": "DONE", "xid": "399",
                    "prior_xids": ["398"]}], legacy=old_legacy()), True),
      ("queue entry naming the (relative) stagedir path directly",
       dict(queue=[{"job_id": "q2", "state": "FAILED",
                    "launch_kwargs": {"stagedir": rel("eqr_run_ca_unref1")}}]),
       False),
      ("control: queue entry naming the stagedir under a non-CitC root",
       dict(queue=[{"job_id": "q2", "state": "FAILED",
                    "workdir": "/x/" + rel("eqr_run_ca_unref1")}]), True),
      ("active registry entry, status COMPLETED",
       dict(jobs={"300000002": {"status": "COMPLETED",
                                "stagedir": rel("eqr_run_ca_unref1")}}), False),
      ("active registry entry, status CANCELLED, absolute path in A",
       dict(jobs={"300000003": {"status": "CANCELLED", "stagedir": "ABS"}}), False),
      ("control: active registry entry, absolute path in ANOTHER client",
       dict(jobs={"300000004": {"status": "SUBMITTED", "stagedir": "ABS_OTHER"}}),
       True),
      ("legacy record archived 5 days ago", dict(legacy=old_legacy(5)), False),
      ("control: legacy record archived 60 days ago", dict(legacy=old_legacy(60)),
       True),
      ("legacy record with no timestamp in a fresh file",
       dict(legacy={XID_OLD: {"status": "COMPLETED",
                              "stagedir": rel("eqr_run_ca_aaa111")}}), False),
      ("control: legacy record with no timestamp in a file untouched for 40 days",
       dict(legacy={XID_OLD: {"status": "COMPLETED",
                              "stagedir": rel("eqr_run_ca_aaa111")}},
            legacy_mtime_age_d=40), True),
      ("provenance record 5 days ago (relative only)",
       dict(prov=[{"ts": iso_ago(5) + "Z", "rel": rel("eqr_run_ca_unref1")}]),
       False),
  ]
  for label, recs, want in cases:
    with Env() as e:
      base_family(e)
      for k in ("jobs",):
        if k in recs:
          for v in recs[k].values():
            if v.get("stagedir") == "ABS":
              v["stagedir"] = os.path.join(e.root, A, "google3",
                                           rel("eqr_run_ca_unref1"))
            elif v.get("stagedir") == "ABS_OTHER":
              v["stagedir"] = os.path.join(e.root, N, "google3",
                                           rel("eqr_run_ca_unref1"))
      e.write_records(**recs)
      _, m = verdict(A)
      _ck(m["retirable"] == want, f"rule 4: {label} -> retirable={want}"
          + ("" if m["retirable"] == want else f" (reasons {m['reasons']})"))

  # An unreadable record source makes the plan incomplete: nothing retirable.
  with Env() as e:
    base_family(e)
    e.write_records(legacy=old_legacy())
    with open(e.legacy, "w") as f:
      f.write('{"truncated": ')
    p, m = verdict(A)
    _ck(not p["complete"] and not m["retirable"],
        "an unreadable legacy registry => plan INCOMPLETE, nothing retirable")
    buf = io.StringIO()
    rc = ret.execute(ret.cfg(), p, out=lambda s: buf.write(s + "\n"))
    _ck(rc == 1 and "REFUSED" in buf.getvalue()
        and not any(x.startswith("g4 citc -d") for x in e.events_list()),
        "execute refuses on an incomplete plan and deletes nothing")


def test_manual_excluded():
  print("test_manual_excluded")
  manual = ["cppflag_probe_0927", "tpu_utils_router_cl", "hg_recovery_20260906",
            "lyy_stage_cl952849434_b", "tpu_staging_20260921",
            "tpu_staging_t215_2026110", "tpu_staging_t215_20261001_x",
            "tpu_staging_lyy_t215_", "run_amply_workspace"]
  with Env() as e:
    base_family(e)
    for n in manual:
      e.manual(n)
    e.member(LA, int(10.5 * MIB), {"eqr_run_ca_lll111": 45})
    e.member(LN, MIB, {"eqr_run_ca_lln111": 0.2})
    e.write_records(legacy=old_legacy())
    p = ret.plan(ret.cfg())
    names = {m["name"] for m in p["members"]}
    _ck(sorted(p["manual"]) == sorted(manual),
        "every non-family client (incl. look-alike names) is listed as manual")
    _ck(not names & set(manual), "no manual client is ever a member/candidate")
    ret_names = sorted(m["name"] for m in p["members"] if m["retirable"])
    _ck(ret_names == [LA, A] or ret_names == sorted([A, LA]),
        f"both families are planned independently: retirable={ret_names}")
    rep = ret.format_report(ret.cfg(), p)
    _ck("manual, not managed (never considered): 9:" in rep,
        "the report lists the manual clients separately")
    buf = io.StringIO()
    rc = ret.execute(ret.cfg(), p, out=lambda s: buf.write(s + "\n"))
    dels = [x for x in e.events_list() if x.startswith("g4 citc -d")]
    _ck(rc == 0 and sorted(dels) == sorted([f"g4 citc -d {A}", f"g4 citc -d {LA}"]),
        f"execute deletes only family candidates: {dels}")
    _ck(all(os.path.isdir(os.path.join(e.root, n)) for n in manual),
        "every manual client still exists after execute")
    buf = io.StringIO()
    rc = ret.execute(ret.cfg(), ret.plan(ret.cfg()), only=["cppflag_probe_0927"],
                     out=lambda s: buf.write(s + "\n"))
    _ck(rc == 1 and "not managed" in buf.getvalue(),
        "execute --client <manual client> is refused")


def test_archive_then_delete_and_restore():
  print("test_archive_then_delete_and_restore")
  with Env() as e:
    base_family(e)
    e.write_records(legacy={**old_legacy(),
                            "300000009": {"status": "CANCELLED",
                                          "stagedir": rel("eqr_run_ca_aaa111"),
                                          "cancelled_at": iso_ago(45)}})
    orig = os.path.join(e.root, A, "google3", rel("eqr_run_ca_aaa111"),
                        "pkg", "main.py")
    with open(orig) as f:
      orig_text = f.read()
    c = ret.cfg()
    p = ret.plan(c)
    buf = io.StringIO()
    rc = ret.execute(c, p, out=lambda s: buf.write(s + "\n"))
    ev = e.events_list()
    _ck(rc == 0, f"execute rc=0 ({buf.getvalue().strip()})")
    fu = [i for i, x in enumerate(ev) if x.startswith("fileutil ")]
    dl = [i for i, x in enumerate(ev) if x.startswith("g4 citc -d")]
    _ck(len(dl) == 1 and fu and max(fu) < dl[0],
        "every fileutil call (tarball, read-back, INDEX) precedes `g4 citc -d`")
    _ck(any(x.startswith("fileutil cp -f") and x.endswith("INDEX.jsonl")
            and "/cns/" in x.split()[-1] for x in ev[:dl[0]]),
        "INDEX.jsonl is uploaded before the delete")
    pg = [i for i, x in enumerate(ev) if x.startswith("pgrep")]
    _ck(all(any(j == i - 1 or (j < i and ev[j].startswith("pgrep")
                               and not any(ev[k].startswith("fileutil")
                                           for k in range(j + 1, i)))
                for j in pg) for i in fu),
        "every fileutil call is preceded by its own pgrep gate")
    _ck(not os.path.exists(os.path.join(e.root, A)), f"{A} was deleted")
    tb = os.path.join(e.cns, "zz-d/home/qiaos/eqr_data/stage_ws_retired", A,
                      "eqr_run_ca_aaa111.tar.gz")
    _ck(os.path.isfile(tb), "the tarball is on (fake) CNS")
    idx_cns = os.path.join(os.path.dirname(tb), "INDEX.jsonl")
    rows = [json.loads(x) for x in open(idx_cns)] if os.path.isfile(idx_cns) else []
    _ck(sorted(r["xid"] for r in rows) == [XID_OLD, "300000009"]
        and all(r["tarball"].endswith(f"{A}/eqr_run_ca_aaa111.tar.gz")
                and r["baseline_cl"] == "989229158"
                and r["stagedir_rel"] == rel("eqr_run_ca_aaa111") for r in rows),
        "CNS INDEX.jsonl maps each XID -> tarball, client, CL, stagedir")
    with open(tb, "rb") as f:
      sha = hashlib.sha256(f.read()).hexdigest()
    _ck(all(r["sha256"] == sha for r in rows), "index sha256 matches the tarball")
    local_rows = [json.loads(x) for x in open(c["index"])]
    _ck(len(local_rows) == 2, "the local index got the same rows")
    _ck(not os.path.exists(os.path.join(e.cns, "zz-d/home/qiaos/eqr_data/"
                                        "stage_ws_retired", A,
                                        "eqr_run_ca_unref1.tar.gz")),
        "an unreferenced stagedir is not archived")
    _ck(os.path.isdir(os.path.join(e.root, N)), "the newest member is untouched")

    # restore --dry-run changes nothing; restore recreates the client.
    buf = io.StringIO()
    rc = ret.restore(c, xid=XID_OLD, dry_run=True, out=lambda s: buf.write(s + "\n"))
    _ck(rc == 0 and f"would run: {e.env['TPU_STAGE_WS_G4']} citc -c 989229158 {A}"
        in buf.getvalue() and not os.path.exists(os.path.join(e.root, A)),
        "restore --dry-run prints the g4 citc -c command and creates nothing")
    buf = io.StringIO()
    rc = ret.restore(c, xid=XID_OLD, out=lambda s: buf.write(s + "\n"))
    got = open(orig).read() if os.path.isfile(orig) else None
    _ck(rc == 0 and got == orig_text,
        "restore --xid recreates the client at its CL and untars the stagedir "
        "byte-identically at the original path")
    _ck(rot.baseline_cl(os.path.join(e.root, A)) == "989229158",
        "the recreated client reads back the recorded baseline CL")
    buf = io.StringIO()
    rc = ret.restore(c, xid="300000009", out=lambda s: buf.write(s + "\n"))
    _ck(rc == 0 and "already present" in buf.getvalue(),
        "a second restore of the same stagedir is a no-op")
    into = os.path.join(e.tmp, "other_ws", "google3")
    os.makedirs(into)
    rc = ret.restore(c, xid=XID_OLD, into=into, out=lambda s: None)
    _ck(rc == 0 and os.path.isfile(os.path.join(into, rel("eqr_run_ca_aaa111"),
                                                "pkg", "main.py")),
        "restore --into an existing google3 root works too")
    rc = ret.restore(c, xid="123", out=lambda s: None)
    _ck(rc == 1, "restore of an unknown XID fails")


def test_execute_refuses_on_archive_failure():
  print("test_execute_refuses_on_archive_failure")
  fail_tar = "#!/bin/bash\necho 'simulated tar failure' >&2\nexit 2\n"
  modes = [("fileutil mkdir fails", {"FAKE_FU_MODE": "fail_mkdir"}),
           ("tarball upload fails", {"FAKE_FU_MODE": "fail_cp_tar"}),
           ("tarball corrupted on CNS (read-back sha256 differs)",
            {"FAKE_FU_MODE": "corrupt"}),
           ("read-back fails", {"FAKE_FU_MODE": "fail_readback"}),
           ("INDEX.jsonl upload fails", {"FAKE_FU_MODE": "fail_index"}),
           ("pgrep gate never clears (> 12 fileutil forever)", {"PGREP": "99"}),
           ("pgrep itself fails", {"FAKE_PGREP_MODE": "error"}),
           ("tar fails", {"TAR": fail_tar}),
           ("archive root unresolved", {"TPU_STAGE_WS_ARCHIVE_ROOT": "/tmp/x"})]
  for label, extra in modes:
    env_extra = {k: v for k, v in extra.items() if k not in ("PGREP", "TAR")}
    with Env(**env_extra) as e:
      if "PGREP" in extra:
        with open(e.pgrep_seq, "w") as f:
          f.write(extra["PGREP"] + "\n")
      if "TAR" in extra:
        os.environ["TPU_STAGE_WS_TAR"] = _script(os.path.join(e.tmp, "tar"),
                                                 extra["TAR"])
      base_family(e)
      e.write_records(legacy=old_legacy())
      c = ret.cfg()
      p = ret.plan(c)
      _ck(next(m for m in p["members"] if m["name"] == A)["retirable"],
          f"[{label}] precondition: {A} is retirable")
      buf = io.StringIO()
      rc = ret.execute(c, p, out=lambda s: buf.write(s + "\n"))
      dels = [x for x in e.events_list() if x.startswith("g4 citc -d")]
      _ck(rc == 1 and not dels and os.path.isdir(os.path.join(e.root, A))
          and "REFUSED" in buf.getvalue(),
          f"[{label}] execute refuses: rc=1, no `g4 citc -d`, {A} intact")
      if "TAR" in extra:
        os.environ.pop("TPU_STAGE_WS_TAR", None)

  # A change between archiving and deleting aborts the delete.
  with Env() as e:
    base_family(e)
    e.write_records(legacy=old_legacy())
    c = ret.cfg()
    p = ret.plan(c)
    m = next(x for x in p["members"] if x["name"] == A)
    ok, _, _ = ret.archive_client(c, m)
    e.write_records(legacy=old_legacy(),
                    queue=[{"job_id": "new", "xid": XID_OLD, "state": "QUEUED"}])
    ok2, detail = ret.delete_client(c, m)
    _ck(ok and not ok2 and "no longer retirable" in detail
        and os.path.isdir(os.path.join(e.root, A)),
        "a queue entry appearing after archiving blocks the delete (re-plan)")
  # The delete waits for the wrapper's stage lock and gives up if it is held.
  with Env(TPU_STAGE_WS_STAGE_LOCK_WAIT_S="0.5") as e:
    base_family(e)
    e.write_records(legacy=old_legacy())
    c = ret.cfg()
    holder = subprocess.Popen(["flock", c["stage_lock"], "sleep", "5"])
    time.sleep(0.3)
    try:
      p = ret.plan(c)
      buf = io.StringIO()
      rc = ret.execute(c, p, out=lambda s: buf.write(s + "\n"))
    finally:
      holder.kill()
      holder.wait()
    _ck(rc == 1 and "stage lock" in buf.getvalue()
        and os.path.isdir(os.path.join(e.root, A)),
        "a held stage lock (a stage in flight) prevents the delete")


def test_fileutil_gate():
  print("test_fileutil_gate")
  with Env() as e:
    with open(e.pgrep_seq, "w") as f:
      f.write("13\n14\n5\n")
    c = ret.cfg()
    t0 = time.time()
    _ck(ret.fileutil_gate(c), "gate opens once pgrep drops to <= 12")
    _ck(sum(1 for x in e.events_list() if x.startswith("pgrep")) == 3,
        "the gate re-checked after each wait (3 pgrep calls)")
    with open(e.pgrep_seq, "w") as f:
      f.write("12\n")
    _ck(ret.fileutil_gate(c), "12 running is allowed (the limit is > 12)")
    with open(e.pgrep_seq, "w") as f:
      f.write("13\n")
    _ck(not ret.fileutil_gate(c), "gate gives up after the max wait while > 12")
    ok, _ = ret.fileutil(c, ["mkdir", "-p", "/cns/zz-d/x"])
    _ck(not ok and not any(x.startswith("fileutil") for x in e.events_list()),
        "fileutil() never runs fileutil when the gate is closed")
    _ck(time.time() - t0 < 10, "gate test ran quickly")


def test_daily_check():
  print("test_daily_check")
  with Env(TPU_STAGE_WS_CLIENT_WARN="3") as e:
    base_family(e)
    for n in ("cppflag_probe_0927", "hg_recovery_20260906", "lyy_stage_x"):
      e.manual(n)
    e.write_records(legacy=old_legacy())
    c = ret.cfg()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
      r1 = ret.daily_check()
    _ck(r1 == "started" and "WARNING: 5 CitC clients" in err.getvalue(),
        "first check: WARNING printed when clients (5) > limit (3)")
    err2 = io.StringIO()
    with contextlib.redirect_stderr(err2):
      r2 = ret.daily_check()
    _ck(r2 == "skip" and not err2.getvalue(),
        "second check within 24 h is skipped silently")
    deadline = time.time() + 20
    log = ""
    while time.time() < deadline:
      log = open(c["log"]).read() if os.path.exists(c["log"]) else ""
      if "CANDIDATE" in log:
        break
      time.sleep(0.2)
    _ck(f"retire-check: CANDIDATE {A}" in log and "clients=5" in log,
        "the detached report logs the candidate and the client count")
    _ck("WARNING: 5 CitC clients" in log, "the WARNING is logged too")
    time.sleep(0.5)
    _ck(not any(x.startswith(("g4 citc -d", "fileutil")) for x in e.events_list())
        and os.path.isdir(os.path.join(e.root, A)),
        "the daily check never archives or deletes (no fileutil, no g4 -d)")
    os.utime(c["stamp"], (time.time() - 2 * DAY,) * 2)
    _ck(ret.daily_check() == "started", "a stamp older than 24 h re-arms the check")
  with Env(TPU_STAGE_WS_CLIENT_WARN="60") as e:
    base_family(e)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
      ret.daily_check()
    _ck("WARNING" not in err.getvalue(), "no WARNING at or below the limit")
  with Env(TPU_STAGE_WS_RETIRE_CHECK="0") as e:
    _ck(ret.daily_check() == "disabled" and not os.path.exists(ret.cfg()["stamp"]),
        "TPU_STAGE_WS_RETIRE_CHECK=0 disables it")
  # Wiring: ensure (non-dry-run) runs it; a dry run does not.
  with Env() as e:
    base_family(e)
    c = ret.cfg()
    with contextlib.redirect_stderr(io.StringIO()), \
        contextlib.redirect_stdout(io.StringIO()):
      rot.ensure(rot._cfg(), "qiaos", dry_run=True)
    _ck(not os.path.exists(c["stamp"]), "ensure --dry-run does not run the check")
    with contextlib.redirect_stderr(io.StringIO()), \
        contextlib.redirect_stdout(io.StringIO()):
      rc = rot.ensure(rot._cfg(), "qiaos")
    _ck(rc == 0 and os.path.exists(c["stamp"]), "ensure runs the daily check")
    time.sleep(1.0)  # let the detached child finish before the temp dir goes


def test_cli():
  print("test_cli")
  with Env() as e:
    base_family(e)
    e.manual("cppflag_probe_0927")
    e.write_records(legacy=old_legacy())
    here = os.path.dirname(os.path.abspath(__file__))
    p = subprocess.run([sys.executable, os.path.join(here, "stage_ws_rotate.py"),
                        "retire"], capture_output=True, text=True, timeout=60)
    _ck(p.returncode == 0 and f"RETIRABLE: {A}" in p.stdout
        and "REPORT-ONLY" in p.stdout and "cppflag_probe_0927" in p.stdout,
        "`stage_ws_rotate.py retire` prints the report")
    _ck(not any(x.startswith(("g4", "fileutil")) for x in e.events_list())
        and os.path.isdir(os.path.join(e.root, A)),
        "report mode calls neither g4 nor fileutil")
    p = subprocess.run([sys.executable, os.path.join(here, "stage_ws_rotate.py"),
                        "retire", "--json"], capture_output=True, text=True,
                       timeout=60)
    d = json.loads(p.stdout) if p.returncode == 0 else {}
    _ck([m["name"] for m in d.get("members", []) if m["retirable"]] == [A],
        "`retire --json` is machine-readable")
    p = subprocess.run([sys.executable, os.path.join(here, "stage_ws_rotate.py"),
                        "retire", "--client", A], capture_output=True, text=True,
                       timeout=60)
    _ck(p.returncode != 0 and os.path.isdir(os.path.join(e.root, A)),
        "--client without --execute is rejected")


def test_default_archive_root():
  print("test_default_archive_root")
  here = os.path.dirname(os.path.abspath(ret.__file__))
  if not os.path.isfile(os.path.join(here, "google3_tpu_utils", "cell_locality.py")):
    print("  skip (no google3_tpu_utils/cell_locality.py next to the module)")
    return
  r = ret.default_archive_root("cmh")
  _ck(r.startswith("/cns/") and r.endswith("/home/qiaos/eqr_data/stage_ws_retired"),
      f"cmh resolves through cell_locality: {r}")
  _ck(ret.default_archive_root("no-such-metro") == "",
      "an unknown metro resolves to '' (execute would refuse)")


# --------------------------------------------------------------------------
# rule 4, relaxed (operator 2026-11-09): a queue / active-registry entry stops
# blocking only when it is terminal AND ended more than 30 days ago.
# --------------------------------------------------------------------------
def ago(days):
  return time.time() - days * DAY


def reg_ts(days):
  """registry_store.cancel's format: local time, naive (the host is UTC)."""
  return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ago(days)))


def qrow(job_id, state, xid=None, prior=(), ended_d=None, start_d=None,
         stagedir=None, sub_state=None, prior_d=None):
  """A queue row shaped like route_lib's (schema v2): prior attempts are
  SUPERSEDED with their own ended_at; the final attempt has ended_at only when
  ended_d is given."""
  start = start_d if start_d is not None else (ended_d or 0) + 1
  pd = prior_d if prior_d is not None else start + 1  # prior attempts ended
  subs = []
  for i, x in enumerate(prior):
    subs.append({"seq": i + 1, "xid": x, "state": "SUPERSEDED",
                 "created_at": ago(pd + 1), "confirmed_at": ago(pd + 1),
                 "ended_reason": "rerouted", "ended_at": ago(pd)})
  if xid:
    s = {"seq": len(subs) + 1, "xid": xid, "state": sub_state or state,
         "created_at": ago(start), "confirmed_at": ago(start)}
    if ended_d is not None:
      s["ended_at"] = ago(ended_d)
    subs.append(s)
  row = {"job_id": job_id, "name": job_id, "state": state, "xid": xid,
         "prior_xids": list(prior), "submissions": subs,
         "submitted_at": ago(start), "build_started_at": None,
         "last_known_checkpoint_checked_at": None,
         "cooldown_pairs": {"gl|h100": {"until": time.time() + 9 * DAY}},
         "launch_kwargs": {"exp_name": job_id}}
  if stagedir:
    row["launch_kwargs"]["stagedir"] = stagedir
  return row


def write_journal(e, lines):
  """<registry>.journal lines shaped like registry_store._journal's."""
  with open(e.jobs + ".journal", "w") as f:
    for days, who, added, changed in lines:
      f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                              time.gmtime(ago(days))),
                          "who": who, "pid": 1, "argv0": "-", "added": added,
                          "removed": [], "changed": changed},
                         sort_keys=True) + "\n")


U1 = "eqr_run_ca_unref1"
XQ = "399000001"   # a queue XID no registry knows (the queue names the path)
X1 = "300000101"   # an old terminal XID (the start of a resume chain)
X2 = "300000102"   # the live resume of X1


def test_relaxed_rule():
  print("test_relaxed_rule")
  cases = [
      # -- queue entries (the entry names the stagedir directly) --
      ("queue DONE, final attempt ended 31d ago",
       dict(queue=[qrow("q1", "DONE", XQ, ended_d=31, stagedir=rel(U1))]), True),
      ("control: queue DONE, final attempt ended 29d ago",
       dict(queue=[qrow("q1", "DONE", XQ, ended_d=29, stagedir=rel(U1))]), False),
      ("queue FAILED (a cancel), ended 31d ago",
       dict(queue=[qrow("q1", "FAILED", XQ, ended_d=31, stagedir=rel(U1),
                        sub_state="CANCELLED")]), True),
      ("queue RUNNING, every timestamp 400d old",
       dict(queue=[qrow("q1", "RUNNING", XQ, start_d=400, stagedir=rel(U1))]),
       False),
      ("control: the same entry DONE, ended 400d ago",
       dict(queue=[qrow("q1", "DONE", XQ, ended_d=400, stagedir=rel(U1))]), True),
      ("queue SUBMITTED / QUEUED / BUDGET_DEFERRED / HELD, 400d old",
       dict(queue=[qrow("q1", "SUBMITTED", XQ, start_d=400, stagedir=rel(U1)),
                   qrow("q2", "QUEUED", None, start_d=400, stagedir=rel(U1)),
                   qrow("q3", "BUDGET_DEFERRED", None, start_d=400,
                        stagedir=rel(U1)),
                   qrow("q4", "HELD", None, start_d=400, stagedir=rel(U1))]),
       False),
      ("queue entry of an UNKNOWN state ('FINISHED'), ended 400d ago",
       dict(queue=[qrow("q1", "FINISHED", XQ, ended_d=400, stagedir=rel(U1))]),
       False),
      ("queue entry with NO state, ended 400d ago",
       dict(queue=[{k: v for k, v in qrow("q1", "DONE", XQ, ended_d=400,
                                          stagedir=rel(U1)).items()
                    if k != "state"}]), False),
      ("queue DONE whose final attempt has no ended_at (older ones do, 400d)",
       dict(queue=[qrow("q1", "DONE", XQ, prior=["399000002"], start_d=400,
                        stagedir=rel(U1))]), False),
      ("queue FAILED with no attempt at all (no timestamp)",
       dict(queue=[{"job_id": "q1", "state": "FAILED", "submissions": [],
                    "launch_kwargs": {"stagedir": rel(U1)}}]), False),
      # -- active-registry rows --
      ("registry CANCELLED, cancelled_at 31d ago",
       dict(jobs={"300000201": {"status": "CANCELLED", "stagedir": rel(U1),
                                "cancelled_at": reg_ts(31)}}), True),
      ("control: registry CANCELLED, cancelled_at 29d ago",
       dict(jobs={"300000201": {"status": "CANCELLED", "stagedir": rel(U1),
                                "cancelled_at": reg_ts(29)}}), False),
      ("registry SUBMITTED, 400d-old timestamp",
       dict(jobs={"300000201": {"status": "SUBMITTED", "stagedir": rel(U1),
                                "cancelled_at": reg_ts(400)}}), False),
      ("registry RUNNING, 400d-old timestamp",
       dict(jobs={"300000201": {"status": "RUNNING", "stagedir": rel(U1),
                                "cancelled_at": reg_ts(400)}}), False),
      ("registry row of an UNKNOWN status ('STOPPED'), 400d-old timestamp",
       dict(jobs={"300000201": {"status": "STOPPED", "stagedir": rel(U1),
                                "cancelled_at": reg_ts(400)}}), False),
      ("registry COMPLETED with no timestamp anywhere",
       dict(jobs={"300000201": {"status": "COMPLETED", "stagedir": rel(U1)}}),
       False),
      ("registry COMPLETED; the queue recorded its end 31d ago",
       dict(jobs={"300000201": {"status": "COMPLETED", "stagedir": rel(U1)}},
            queue=[qrow("q1", "DONE", "300000201", ended_d=31)]), True),
      ("control: registry COMPLETED; the queue recorded its end 29d ago",
       dict(jobs={"300000201": {"status": "COMPLETED", "stagedir": rel(U1)}},
            queue=[qrow("q1", "DONE", "300000201", ended_d=29)]), False),
      ("registry TERMINAL_RECONCILED / TERMINAL_NOT_RUNNING; queue end 31d",
       dict(jobs={"300000201": {"status": "TERMINAL_RECONCILED",
                                "stagedir": rel(U1)},
                  "300000202": {"status": "TERMINAL_NOT_RUNNING",
                                "stagedir": rel(U1)}},
            queue=[qrow("q1", "DONE", "300000201", ended_d=31),
                   qrow("q2", "DONE", "300000202", ended_d=40)]), True),
  ]
  journal_cases = [
      ("registry COMPLETED; journal saw SUBMITTED->COMPLETED 31d ago",
       [(35, "xm_launcher", ["300000201"], {}),
        (35, "tpu queue (register)", [], {"300000201": {"status": [None, "SUBMITTED"]}}),
        (31, "tpu_check_daemon (status pass)", [],
         {"300000201": {"status": ["SUBMITTED", "COMPLETED"]}})], True),
      ("control: ... journal saw it 29d ago",
       [(35, "xm_launcher", ["300000201"], {}),
        (29, "tpu_check_daemon (status pass)", [],
         {"300000201": {"status": ["SUBMITTED", "COMPLETED"]}})], False),
      ("registry COMPLETED; journal's newest status change is ->SUBMITTED "
       "(40d): the move to COMPLETED was unjournaled, time unknown",
       [(40, "xm_launcher", ["300000201"], {}),
        (40, "tpu queue (register)", [], {"300000201": {"status": [None, "SUBMITTED"]}})],
       False),
      ("registry COMPLETED; journal says COMPLETED 40d ago but touched the row "
       "again 5d ago (newest line counts)",
       [(40, "tpu_check_daemon (status pass)", [],
         {"300000201": {"status": ["SUBMITTED", "COMPLETED"]}}),
        (5, "tpu_check_daemon (status pass)", [],
         {"300000201": {"error": ["", "x"]}})], False),
  ]
  for label, recs, want in cases:
    with Env() as e:
      base_family(e)
      e.write_records(**recs)
      _, m = verdict(A)
      _ck(m["retirable"] == want, f"relaxed: {label} -> retirable={want}"
          + ("" if m["retirable"] == want else f" (reasons {m['reasons']})"))
  for label, lines, want in journal_cases:
    with Env() as e:
      base_family(e)
      e.write_records(jobs={"300000201": {"status": "COMPLETED",
                                          "stagedir": rel(U1)}})
      write_journal(e, lines)
      _, m = verdict(A)
      _ck(m["retirable"] == want, f"relaxed: {label} -> retirable={want}"
          + ("" if m["retirable"] == want else f" (reasons {m['reasons']})"))

  # TPU_STAGE_WS_RETIRE_ENDED_DAYS moves the cut.
  with Env(TPU_STAGE_WS_RETIRE_ENDED_DAYS="45") as e:
    base_family(e)
    e.write_records(queue=[qrow("q1", "DONE", XQ, ended_d=40, stagedir=rel(U1))])
    _, m = verdict(A)
    _ck(not m["retirable"], "relaxed: with ENDED_DAYS=45 an entry ended 40d ago blocks")

  # An unreadable journal makes the plan incomplete.
  with Env() as e:
    base_family(e)
    e.write_records(jobs={"300000201": {"status": "CANCELLED", "stagedir": rel(U1),
                                        "cancelled_at": reg_ts(60)}})
    p, m = verdict(A)
    _ck(m["retirable"], "journal control: without a journal the row is old -> retirable")
    os.makedirs(e.jobs + ".journal")  # exists, cannot be read as a file
    p, m = verdict(A)
    _ck(not p["complete"] and not m["retirable"]
        and any(".journal" in x for x in p["errors"]),
        "an unreadable registry journal => plan INCOMPLETE, nothing retirable")


def test_relaxed_chain():
  print("test_relaxed_chain")
  # X1 is old and terminal (its own row does not block); X2 resumes it.
  old_x1 = {X1: {"status": "CANCELLED", "stagedir": rel("eqr_run_ca_aaa111"),
                 "cancelled_at": reg_ts(400)}}
  cases = [
      ("control: X1 alone (CANCELLED 400d ago) does not block",
       dict(jobs=old_x1), True),
      ("a RUNNING queue entry lists X1 in prior_xids",
       dict(jobs=old_x1, queue=[qrow("r1", "RUNNING", X2, prior=[X1],
                                     start_d=1, prior_d=400)]), False),
      ("control: the same entry DONE, ended 400d ago",
       dict(jobs=old_x1, queue=[qrow("r1", "DONE", X2, prior=[X1],
                                     ended_d=400, start_d=401,
                                     prior_d=402)]), True),
      ("a queue entry DONE only 5d ago lists X1 in prior_xids",
       dict(jobs=old_x1, queue=[qrow("r1", "DONE", X2, prior=[X1],
                                     ended_d=5, start_d=6, prior_d=400)]),
       False),
      ("a RUNNING entry lists X1 only in submissions[].xid",
       dict(jobs=old_x1,
            queue=[{**qrow("r1", "RUNNING", X2, prior=[X1], start_d=1,
                           prior_d=400),
                    "prior_xids": []}]), False),
      ("a RUNNING entry names X1 as its xid (resume re-used the XID's row)",
       dict(jobs=old_x1, queue=[qrow("r1", "RUNNING", X1, start_d=1)]), False),
      ("X1 resolved through the LEGACY registry, RUNNING entry lists it",
       dict(legacy={X1: {"status": "COMPLETED",
                         "stagedir": rel("eqr_run_ca_aaa111"),
                         "archived_at": iso_ago(400)}},
            queue=[qrow("r1", "RUNNING", X2, prior=[X1], start_d=1,
                        prior_d=400)]), False),
      ("an active SUBMITTED row lists X1 as resume_xid",
       dict(jobs={**old_x1, X2: {"status": "SUBMITTED", "resume_xid": X1,
                                 "stagedir": rel("eqr_run_ca_zzz999")}}), False),
      ("control: an active CANCELLED-400d row lists X1 as resume_xid",
       dict(jobs={**old_x1, X2: {"status": "CANCELLED", "resume_xid": X1,
                                 "cancelled_at": reg_ts(400),
                                 "stagedir": rel("eqr_run_ca_zzz999")}}), True),
  ]
  for label, recs, want in cases:
    with Env() as e:
      base_family(e)
      e.write_records(**recs)
      _, m = verdict(A)
      _ck(m["retirable"] == want, f"chain: {label} -> retirable={want}"
          + ("" if m["retirable"] == want else f" (reasons {m['reasons']})"))
      if not want:
        blk = m["blocking"].get("eqr_run_ca_aaa111", [])
        _ck(any(x.xid == X1 for x in blk),
            f"chain: ... the blocking reference is X1's ({[x.describe() for x in blk]})")


def test_relaxed_counts_and_report():
  print("test_relaxed_counts_and_report")
  with Env() as e:
    base_family(e)
    e.write_records(
        queue=[qrow("a", "DONE", "399000011", ended_d=40),
               qrow("b", "FAILED", "399000012", ended_d=35),
               qrow("c", "DONE", "399000013", ended_d=3),
               qrow("d", "RUNNING", "399000014", start_d=50),
               qrow("e", "DONE", "399000015", start_d=50),   # no ended_at
               qrow("f", "WEIRD", "399000016", ended_d=50)],
        jobs={"300000301": {"status": "COMPLETED", "stagedir": rel(U1)},
              "300000302": {"status": "CANCELLED", "stagedir": rel(U1),
                            "cancelled_at": reg_ts(60)},
              "300000303": {"status": "SUBMITTED", "stagedir": rel(U1)}})
    c = ret.cfg()
    p = ret.plan(c)
    rc = p["records"].entry_counts
    _ck(rc.get(e.queue) == {"ended-old": 2, "recent": 1, "live": 2, "no-end-ts": 1},
        f"per-file entry counts, queue: {rc.get(e.queue)}")
    _ck(rc.get(e.jobs) == {"no-end-ts": 1, "ended-old": 1, "live": 1},
        f"per-file entry counts, registry: {rc.get(e.jobs)}")
    rep = ret.format_report(c, p)
    _ck("terminal & ended >30d: 2 (no block); blocking: 4 = 2 live/unknown "
        "status + 1 terminal without end timestamp + 1 terminal ended <=30d" in rep,
        "the report prints the per-file terminal-old vs blocking counts")
    _ck("ended <= 30d ago (terminal = queue DONE/FAILED, registry CANCELLED/"
        "COMPLETED/FAILED/TERMINAL_NOT_RUNNING/TERMINAL_RECONCILED)" in rep,
        "the report's rules line states the relaxed rule and the terminal sets")
    m = next(x for x in p["members"] if x["name"] == A)
    _ck(any("COMPLETED, no end timestamp" in x.describe()
            for x in m["blocking"].get(U1, [])),
        "a blocking reference says why it still blocks")
  here = os.path.dirname(os.path.abspath(ret.__file__))
  try:
    sys.path.insert(0, os.path.join(here, "google3_tpu_utils"))
    import route_lib  # pylint: disable=import-outside-toplevel
  except Exception as ex:  # pylint: disable=broad-except
    print(f"  skip route_lib drift check ({ex!r})")
  else:
    _ck({s.value for s in route_lib.FINISHED_STATES} == set(ret.QUEUE_TERMINAL),
        f"QUEUE_TERMINAL == route_lib.FINISHED_STATES "
        f"({sorted(s.value for s in route_lib.FINISHED_STATES)})")
  finally:
    sys.path.pop(0)


def test_relaxed_archive_restore():
  print("test_relaxed_archive_restore")
  X3 = "300000401"
  with Env() as e:
    base_family(e)
    e.write_records(
        jobs={X3: {"status": "CANCELLED", "stagedir": rel("eqr_run_ca_aaa111"),
                   "cancelled_at": reg_ts(45)}},
        queue=[qrow("old1", "FAILED", X3, ended_d=45, start_d=46,
                    sub_state="CANCELLED")])
    orig = os.path.join(e.root, A, "google3", rel("eqr_run_ca_aaa111"),
                        "pkg", "main.py")
    orig_text = open(orig).read()
    c = ret.cfg()
    p = ret.plan(c)
    m = next(x for x in p["members"] if x["name"] == A)
    _ck(m["retirable"] and sorted(m["archive"]) == ["eqr_run_ca_aaa111"]
        and {x.xid for x in m["archive"]["eqr_run_ca_aaa111"]} == {X3},
        "an old terminal stagedir (registry + queue, 45d) is retirable and in "
        "the archive set under its XID")
    buf = io.StringIO()
    rc = ret.execute(c, p, out=lambda s: buf.write(s + "\n"))
    ev = e.events_list()
    fu = [i for i, x in enumerate(ev) if x.startswith("fileutil ")]
    dl = [i for i, x in enumerate(ev) if x.startswith("g4 citc -d")]
    _ck(rc == 0 and len(dl) == 1 and fu and max(fu) < dl[0],
        f"execute archives (every fileutil call) before `g4 citc -d` ({buf.getvalue().strip()})")
    tb = os.path.join(e.cns, "zz-d/home/qiaos/eqr_data/stage_ws_retired", A,
                      "eqr_run_ca_aaa111.tar.gz")
    idx = os.path.join(os.path.dirname(tb), "INDEX.jsonl")
    rows = [json.loads(x) for x in open(idx)] if os.path.isfile(idx) else []
    sha = hashlib.sha256(open(tb, "rb").read()).hexdigest() if os.path.isfile(tb) else ""
    _ck([r["xid"] for r in rows] == [X3] and rows[0]["sha256"] == sha
        and rows[0]["stagedir_rel"] == rel("eqr_run_ca_aaa111"),
        "CNS INDEX.jsonl maps the old terminal XID -> tarball, sha256 verified")
    _ck(not os.path.exists(os.path.join(e.root, A)), f"{A} was deleted after archiving")
    rc = ret.restore(c, xid=X3, out=lambda s: None)
    got = open(orig).read() if os.path.isfile(orig) else None
    _ck(rc == 0 and got == orig_text,
        "restore --xid <old terminal XID> recreates the client and the stagedir "
        "byte-identically")


if __name__ == "__main__":
  for t in (test_rules, test_manual_excluded, test_archive_then_delete_and_restore,
            test_execute_refuses_on_archive_failure, test_fileutil_gate,
            test_daily_check, test_cli, test_default_archive_root,
            test_relaxed_rule, test_relaxed_chain, test_relaxed_counts_and_report,
            test_relaxed_archive_restore):
    t()
  print()
  if _fail:
    print(f"FAILED ({len(_fail)}):")
    for f in _fail:
      print(f"  - {f}")
    sys.exit(1)
  print("ALL PASSED")
