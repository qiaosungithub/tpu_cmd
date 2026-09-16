#!/usr/bin/env python3
"""PROTOTYPE of the build-speed block to embed in `tpu check` (tpu_wrapper.sh).

Reads the local queue JSON and renders:
  * the build currently in flight, with live elapsed seconds (build_started_at);
  * the most-recent completed builds, with their recorded last_build_duration.

Both fields come straight off the queue rows -- no RPC, so it stays instant like
the rest of `tpu check`. last_build_duration is None on rows built by the OLD
binary (predates the field); we fall back to nothing rather than inventing a
number. This file is a throwaway; the verified logic is pasted into the wrapper.
"""
import json, os, sys, time

path = os.environ.get('TPU_LQ_FILE', os.path.expanduser('~/.tpu_local_queue.json'))
try:
    with open(path) as f:
        raw = json.load(f)
except Exception:
    sys.exit(0)
entries = raw.get('entries', raw) if isinstance(raw, dict) else raw
if not entries:
    sys.exit(0)

now = time.time()

def _name(e):
    lk = e.get('launch_kwargs', {}) or {}
    return (lk.get('exp_name') or lk.get('config') or str(e.get('job_id', '?')))[:30]

def _bar(sec, unit=3.0, cap=40):
    """A single-glyph-per-`unit`-seconds bar, capped so a slow outlier can't wrap."""
    n = int(round((sec or 0) / unit))
    return '#' * min(n, cap) + ('+' if n > cap else '')

DIM = '\033[2m'; RST = '\033[0m'; CYAN = '\033[1;36m'; MAG = '\033[1;35m'; GRN = '\033[32m'

# --- current build in flight ---
building = [e for e in entries if e.get('state') == 'BUILDING' and e.get('build_started_at')]
# --- recent completed builds (have a recorded duration), newest first by submit ---
done = [e for e in entries
        if e.get('last_build_duration') is not None and e.get('submitted_at')]
done.sort(key=lambda e: e.get('submitted_at', 0), reverse=True)
recent = done[:6]

if not building and not recent:
    # Nothing to show (e.g. all rows predate the field). Print a one-liner so the
    # operator knows the feature is live but has no data yet, rather than silence.
    print(f"\n{CYAN}== Build speed =={RST}   {DIM}(no per-build timing recorded yet){RST}")
    sys.exit(0)

print(f"\n{CYAN}== Build speed =={RST}")

if building:
    for e in building:
        el = now - e['build_started_at']
        print(f"  {MAG}building now{RST}  {_name(e):30s} {el:5.0f}s {MAG}{_bar(el)}{RST}")

if recent:
    durs = [e['last_build_duration'] for e in recent]
    print(f"  {DIM}last {len(recent)} builds:{RST}")
    for e in recent:
        d = e['last_build_duration']
        print(f"    {_name(e):30s} {d:5.0f}s {GRN}{_bar(d)}{RST}")
    med = sorted(durs)[len(durs)//2]
    print(f"  {DIM}median {med:.0f}s  range {min(durs):.0f}-{max(durs):.0f}s  "
          f"(blaze floor ~40s; end-to-end incl. staging+submit is longer){RST}")
